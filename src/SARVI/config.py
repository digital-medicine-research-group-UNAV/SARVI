from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class AppPaths(BaseModel):
    base_dir: Path
    data_input: Path
    data_intermediate: Path
    data_output: Path
    logs_dir: Path
    docs_dir: Path

    @classmethod
    def from_settings(cls) -> "AppPaths":
        return cls(
            base_dir=BASE_DIR,
            data_input=BASE_DIR / "data" / "input",
            data_intermediate=BASE_DIR / "data" / "intermediate",
            data_output=BASE_DIR / "data" / "output",
            logs_dir=BASE_DIR / "logs",
            docs_dir=BASE_DIR / "docs",
        )

paths = AppPaths.from_settings()
load_dotenv(BASE_DIR / "private" / "data.env")