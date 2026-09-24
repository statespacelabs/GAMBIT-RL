"""Phase 2 data schemas: WindowRecord, TransitionRecord, ActionSchema."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowRecord:
    """
    Represents one 5-second encoder input window.

    A WindowRecord is not an RL transition. It only describes the data needed
    to run the frozen encoder over a fixed 150-frame window.

    Fields:
        window_id:
            Deterministic ID, usually "{session_id}_t{t_end_ms}".

        player_id:
            Original player identifier from the manifest or analytics JSON.

        session_id:
            Continuous gameplay session identifier. Used to prevent split leakage.

        t_start:
            Start time of the 5-second observation window in seconds.

        t_end:
            End time of the 5-second observation window in seconds.

        video_path:
            Path to the rendered gameplay video containing this window.

        tel_json_path:
            Path to raw gameplay analytics JSON.

        tel_npz_path:
            Path to preprocessed telemetry NPZ for this window.

        split:
            Dataset split: train, val, or test.

        source_clip_id:
            Original clip/chunk ID used to trace the window back to source data.

        start_frame:
            Video frame index corresponding to t_start (round(t_start * fps)).

        end_frame:
            Video frame index corresponding to t_end (start_frame + seq_len).
    """

    window_id: str
    player_id: str
    session_id: str
    t_start: float
    t_end: float
    video_path: str
    tel_json_path: str
    tel_npz_path: str
    split: str
    source_clip_id: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class TransitionRecord:
    """
    Represents one offline RL transition.

    A TransitionRecord links two encoder windows that are one second apart.
    The state is the latent for the current 5-second window. The next_state is
    the latent for the shifted 5-second window. The action and reward are
    extracted from the raw JSON interval between the two windows.

    Fields:
        transition_id:
            Deterministic transition ID.

        player_id:
            Player identifier for analysis only. This must not be used as a
            policy input.

        session_id:
            Continuous session ID. state and next_state must belong to the
            same session.

        t:
            Decision time in seconds. state_t represents history ending at t.

        state_window_id:
            Window ID for latent z_t.

        next_state_window_id:
            Window ID for latent z_{t+1}.

        state_latent_path:
            Path to .npz containing z_raw and z_norm for state_t.

        next_state_latent_path:
            Path to .npz containing z_raw and z_norm for next_state.

        action_path:
            Path to .npz containing compact next-1-second action vector.

        reward:
            Scalar reward computed from raw gameplay analytics over [t, t+1].

        done:
            True if this is the final valid transition in a session.

        timeout:
            True if the transition ended because the assessment/session window
            ended rather than because of a terminal game event.

        split:
            train, val, or test.
    """

    transition_id: str
    player_id: str
    session_id: str
    t: float
    state_window_id: str
    next_state_window_id: str
    state_latent_path: str
    next_state_latent_path: str
    action_path: str
    reward: float
    done: bool
    timeout: bool
    split: str


@dataclass(frozen=True)
class ActionSchema:
    """
    Defines the layout of the compact 1-second action vector.

    This class records which action dimensions are continuous and which are
    binary/count-like. It is required so BC and IQL can apply the correct losses,
    normalization, and inference post-processing.

    Fields:
        names:
            Ordered list of action dimension names.

        continuous_indices:
            Indices for continuous values such as move mean or look delta.

        binary_indices:
            Indices for binary values such as shoot_pressed or reload_pressed.

        count_indices:
            Indices for count values such as shoot_count or action_count.
    """

    names: tuple[str, ...]
    continuous_indices: tuple[int, ...]
    binary_indices: tuple[int, ...]
    count_indices: tuple[int, ...]

    @property
    def dim(self) -> int:
        """Total number of action dimensions."""
        return len(self.names)


# ─── Default 14-dimensional action schema from the spec ──────────────────────
DEFAULT_ACTION_SCHEMA = ActionSchema(
    names=(
        "move_x_mean",            # 0  continuous
        "move_y_mean",            # 1  continuous
        "look_dx_sum",            # 2  continuous
        "look_dy_sum",            # 3  continuous
        "look_dx_mean",           # 4  continuous
        "look_dy_mean",           # 5  continuous
        "look_speed_mean",        # 6  continuous
        "shoot_count",            # 7  count
        "shoot_pressed",          # 8  binary
        "reload_pressed",         # 9  binary
        "jump_pressed",           # 10 binary
        "crouch_pressed",         # 11 binary
        "targeted_action_count",  # 12 count
        "action_count",           # 13 count
    ),
    continuous_indices=(0, 1, 2, 3, 4, 5, 6),
    binary_indices=(8, 9, 10, 11),
    count_indices=(7, 12, 13),
)
