from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:  # ✅ Fixed spelling
    trained_file_path: str
    test_file_path: str
