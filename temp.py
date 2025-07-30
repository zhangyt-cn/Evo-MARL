

from datasets import load_dataset


mini_JailBreakV_28K = load_dataset("JailbreakV-28K/JailBreakV-28k", 'JailBreakV_28K')["mini_JailBreakV_28K"]
JailBreakV_28K = load_dataset("JailbreakV-28K/JailBreakV-28k", 'JailBreakV_28K')["JailBreakV_28K"]
RedTeam_2K = load_dataset("JailbreakV-28K/JailBreakV-28k", 'RedTeam_2K')["RedTeam_2K"]
