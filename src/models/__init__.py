from enum import Enum

class ModelType(Enum):
    Feedforward = "feedforward"
    LSTM = "lstm"
    Binary = "binary"

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.name.lower() == value.lower():
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")