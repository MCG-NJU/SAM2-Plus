from typing import List, Tuple, Dict

class CkptCopyKeyKernel:
    """
    Copies the keys in the given model state_dict based on specific patterns,
    adding new key-value pairs while keeping the original ones.

    Args:
        rename_patterns: A list of tuples where each tuple contains two elements:
                         the original prefix and the new prefix to replace the original one.
    """

    def __init__(self, rename_patterns: List[Tuple[str, str]]):
        self.rename_patterns = rename_patterns

    def __call__(self, state_dict: Dict):
        """
        Args:
            state_dict: A dictionary representing the given checkpoint's state dict.
        """
        new_state_dict = state_dict.copy()
        for k, v in state_dict.items():
            for original_prefix, new_prefix in self.rename_patterns:
                if k.startswith(original_prefix):
                    new_key = k.replace(original_prefix, new_prefix)
                    new_state_dict[new_key] = v
        return new_state_dict