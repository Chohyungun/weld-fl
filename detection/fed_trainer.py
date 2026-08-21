"""연합학습용 검출 트레이너 — Ultralytics `DetectionTrainer` 래퍼 (지정 함정 구간 #1).

## 왜 래핑하는가

Ultralytics는 자체 저장·재개·조기 종료·EMA 로직을 갖고 있고, 그 기본 동작 몇 개가 이
연구의 공정성 주장과 정면으로 어긋난다. 선언만 해 두면 프레임워크가 조용히 다르게
동작하므로, 어긋나는 지점을 코드로 막는다.

| 기본 동작 | 어긋나는 규칙 | 처방 |
|---|---|---|
| `final_eval()`이 `best.pt`를 로드해 검증 | last 채점 (best 선택은 암묵적 조기 종료) | no-op |
| `EarlyStopping(patience)` | 조기 종료 금지 | patience를 epoch 이상 + stopper 스텁 |
| `optimizer='auto'`가 `lr0`·`momentum`을 버리고 AdamW로 교체 | 5칸 공통 고정의 '최적화' | `SGD` 명시 + 실사용 기록 |
| `save_model()`이 EMA 가중치를 저장 | FedAvg는 raw 가중치를 평균한다 | EMA 비활성 + no-op |
| mlflow 자동 연동, 전역 `runs_dir` 재지향 | 로깅 단일 경로, 저장소 위생 | `mlflow=False` + `project` 절대경로 |

## 라운드 경계를 어떻게 만드는가

`epochs`는 라운드 길이가 아니라 **전역 예산 N**(=100)으로 고정한다. 그래야 warmup과
cosine 스케줄이 중앙집중 학습과 같은 궤적을 그린다. 라운드 r은 `start_epoch = r × E`에서
시작해 로컬 E epoch만 돌고 멈춘다. 멈추는 방식은 조기 종료가 아니라 **예산 도달 정지**이며,
둘을 로그에서 구분할 수 있게 남긴다.

로컬·중앙 칸도 `R=1, E=100`인 퇴화 케이스로 이 트레이너를 통과시킨다. 세 칸이 서로 다른
학습 루프를 타면 "구조 내 동일" 원칙이 문면으로만 남는다.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    from ultralytics.models.yolo.detect import DetectionTrainer
except ModuleNotFoundError as exc:  # pragma: no cover - 설치 환경에서는 도달하지 않는다
    raise ModuleNotFoundError(
        "detection extra 가 설치되지 않았다. `uv sync --extra detection` 으로 ultralytics 를 "
        "설치해야 이 모듈을 쓸 수 있다. 검출 학습을 돌리지 않는 환경(채점기·좌표 모듈 등)은 "
        "이 모듈을 import 하지 않아도 된다."
    ) from exc

from detection import serialize

__all__ = ["FedDetectionTrainer", "NoEarlyStopping", "RoundBudget"]


class NoEarlyStopping:
    """`EarlyStopping` 자리에 끼우는 스텁. 절대 중단시키지 않고 호출 이력만 남긴다.

    `patience`를 크게 잡는 것만으로도 조기 종료는 막히지만, 그것은 "설정이 그러하다"는
    간접 증거일 뿐이다. 이 스텁은 **중단이 구조적으로 불가능함**을 보장하면서, 동시에
    호출 이력을 남겨 사후 감사에서 "조기 종료가 걸리지 않았다"를 증명하게 해 준다.

    `possible_stop` 속성이 반드시 있어야 한다 — 학습 루프의 검증 게이트
    (`self.args.val or final_epoch or self.stopper.possible_stop or self.stop`)가 이 값을
    참조하므로, 없으면 AttributeError로 학습이 죽는다.
    """

    def __init__(self) -> None:
        self.possible_stop: bool = False
        self.best_fitness = 0.0
        self.calls: list[tuple[int, float | None]] = []

    def __call__(self, epoch: int, fitness: Any = None) -> bool:
        self.calls.append((int(epoch), None if fitness is None else float(fitness)))
        return False


class RoundBudget:
    """로컬 예산(E epoch)에 도달하면 학습 루프를 멈추는 콜백.

    `on_fit_epoch_end` 직후에 `if self.stop: break`가 있으므로 즉시 종료된다.
    조기 종료와 구분하기 위해 발화 시점을 따로 기록한다 — 사후 감사에서 `stop`이 켜진
    이유가 예산인지 조기 종료인지 대조하는 근거다.
    """

    def __init__(self, local_epochs: int) -> None:
        self.local_epochs = int(local_epochs)
        self.fired_at_epoch: int | None = None
        self.epochs_ran: int = 0

    def __call__(self, trainer: Any) -> None:
        self.epochs_ran = trainer.epoch - trainer.start_epoch + 1
        if self.epochs_ran >= self.local_epochs:
            trainer.stop = True
            if self.fired_at_epoch is None:
                self.fired_at_epoch = int(trainer.epoch)


class FedDetectionTrainer(DetectionTrainer):
    """라운드 단위로 호출되는 검출 트레이너.

    내부 접촉점은 아래 넷으로 한정한다. 늘리면 Ultralytics 버전 변동에 그만큼 취약해진다.
      1. `_setup_train`  — 가중치 주입, 전역 epoch 오프셋, mosaic 재적용, EMA off, stopper 교체
      2. `validate`      — no-op
      3. `final_eval`    — no-op (stock은 best.pt를 로드한다)
      4. `save_model`    — no-op (stock은 EMA 가중치를 저장한다)
    """

    def __init__(
        self,
        cfg: Any = None,
        overrides: dict | None = None,
        _callbacks: dict | None = None,
        *,
        weights_in: Sequence[np.ndarray] | None = None,
        canonical_keys: Sequence[str] | None = None,
        round_idx: int = 0,
        local_epochs: int | None = None,
    ) -> None:
        self._weights_in = weights_in
        self._canonical_keys = list(canonical_keys) if canonical_keys is not None else None
        self._round_idx = int(round_idx)
        self._local_epochs = local_epochs
        self.injection_digest: list[float] = []
        self.budget: RoundBudget | None = None

        if cfg is None:
            from ultralytics.cfg import get_cfg  # 지역 import — 모듈 로드를 가볍게 유지
            cfg = get_cfg()
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        if self._local_epochs is not None:
            self.budget = RoundBudget(self._local_epochs)
            # 인스턴스 로컬 등록. 전역 default_callbacks 를 건드리면 라운드마다 콜백이 쌓인다.
            self.add_callback("on_fit_epoch_end", self.budget)

    # -- 접촉점 1 -----------------------------------------------------------
    def _setup_train(self) -> None:
        super()._setup_train()

        # (a) 서버 가중치 주입. strict=True — 키가 하나라도 어긋나면 즉시 실패한다.
        if self._weights_in is not None:
            if self._canonical_keys is None:
                raise ValueError("weights_in을 주려면 canonical_keys도 함께 줘야 한다.")
            target = self._unwrapped_model()
            ref = target.state_dict()
            serialize.assert_compatible(self._weights_in, self._canonical_keys, ref)
            sd = serialize.ndarrays_to_state_dict(self._weights_in, self._canonical_keys, ref)
            target.load_state_dict(sd, strict=True)
            # (b) 주입 증빙. 서버가 보낸 값과 대조해 주입이 무력화되지 않았음을 확인한다.
            after = serialize.state_dict_to_ndarrays(target.state_dict(), self._canonical_keys)
            self.injection_digest = serialize.tensor_digest(after)

        # (c) 전역 epoch 오프셋. epochs는 전역 예산 N이고 start_epoch만 라운드마다 움직인다.
        if self._local_epochs is not None:
            self.start_epoch = self._round_idx * int(self._local_epochs)

        # (d) 스케줄러 재개. super()가 start_epoch=0 기준으로 이미 설정했으므로 다시 맞춘다.
        #     이 한 줄이 연합 칸의 학습률 궤적을 중앙집중 칸과 같게 만든다.
        self.scheduler.last_epoch = self.start_epoch - 1

        # (e) mosaic 종료 재적용. 학습 루프의 조건이 `epoch == epochs - close_mosaic` 등호라
        #     라운드 재진입으로 그 시점을 건너뛰면 영영 발화하지 않는다.
        if self.start_epoch >= self.epochs - self.args.close_mosaic:
            self._close_dataloader_mosaic()
            self.train_loader.reset()

        # (f) EMA 비활성. FedAvg가 평균하는 것은 raw 가중치다.
        if self.ema is not None:
            self.ema.enabled = False

        # (g) 조기 종료를 구조적으로 불가능하게 만든다.
        self.stopper = NoEarlyStopping()
        self.stop = False

    # -- 접촉점 2·3·4 -------------------------------------------------------
    def validate(self) -> tuple[dict, None]:
        """no-op. 클라이언트는 검증하지 않는다.

        `val=False`로 두어도 전역 마지막 epoch에서 `final_epoch`이 참이 되어 검증 게이트가
        열리고, 예산 정지로 `self.stop`이 켜져도 마찬가지다. 라운드별 지표는 서버가 val로
        산출하므로 여기서 도는 검증은 시간만 쓴다.
        """
        return {}, None

    def final_eval(self) -> None:
        """no-op. stock은 `best.pt`를 로드해 검증한다 — best 선택 경로 자체를 제거한다."""
        return None

    def save_model(self) -> bool:
        """no-op. stock 체크포인트는 EMA 가중치를 담는다. 저장은 서버가 맡는다.

        `False`를 돌려주어 `on_model_save` 콜백도 돌지 않게 한다.
        """
        return False

    # -- 보조 ---------------------------------------------------------------
    def _unwrapped_model(self):
        """DDP 래핑을 벗긴 실제 `nn.Module`."""
        return getattr(self.model, "module", self.model)

    def export_weights(self) -> list[np.ndarray]:
        """학습 후 raw 가중치를 정본 키 순서로 내보낸다."""
        if self._canonical_keys is None:
            raise ValueError("canonical_keys가 없다.")
        return serialize.state_dict_to_ndarrays(
            self._unwrapped_model().state_dict(), self._canonical_keys
        )

    def effective_optimizer(self) -> dict[str, Any]:
        """실제로 쓰인 옵티마이저와 학습률.

        `optimizer='auto'`는 명시한 `lr0`·`momentum`을 버리고 AdamW로 갈아치우면서 경고
        한 줄만 남긴다. 설정에 무엇을 적었는지가 아니라 **무엇이 실제로 돌았는지**를
        회계 매트릭스에 남겨야 5칸 공통 고정이 지켜졌음을 사후에 증명할 수 있다.
        """
        opt = self.optimizer
        groups = getattr(opt, "param_groups", [])
        first = groups[0] if groups else {}
        return {
            "optimizer": type(opt).__name__,
            "lr": float(first.get("lr", float("nan"))),
            "momentum": float(first.get("momentum", first.get("betas", [float("nan")])[0]))
            if ("momentum" in first or "betas" in first)
            else float("nan"),
            "arg_optimizer": str(self.args.optimizer),
            "arg_lr0": float(self.args.lr0),
            "arg_momentum": float(self.args.momentum),
        }
