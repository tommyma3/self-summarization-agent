"""Common actor entry point for token-preserving hybrid-model training."""
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker

from self_summarization_agent.qwen_interval_attention import install_qwen_interval_attention


class IntervalActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        install_qwen_interval_attention()
        super().init_model()
        if self.config.get("sparse_policy_head", False):
            from self_summarization_agent.sparse_policy_head import install_sparse_policy_head
            install_sparse_policy_head(self.actor.engine)
        if self.config.get("zero_advantage_forward_only", False):
            from self_summarization_agent.zero_advantage import install_zero_advantage_forward_only
            install_zero_advantage_forward_only(self.actor.engine)
