import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = Path(__file__).resolve().parents[3] / 'src'
REPO_ROOT = PROJECT_ROOT.parents[1]
PROJECT_ROOT_RELATIVE_PARTS = PROJECT_ROOT.relative_to(REPO_ROOT).parts
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from const import CLOUD_BUNDLE_ARCHIVE_PATH, CLOUD_BUNDLE_DIR, RTDFORMER2_CLOUD_BUNDLE_FILES


def resolve_project_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path

    if path.parts[:len(PROJECT_ROOT_RELATIVE_PARTS)] == PROJECT_ROOT_RELATIVE_PARTS:
        stripped_parts = path.parts[len(PROJECT_ROOT_RELATIVE_PARTS):]
        path = Path(*stripped_parts) if stripped_parts else Path()

    return PROJECT_ROOT / path


def validate_bundle_sources(bundle_files):
    missing_files = []
    for relative_path in bundle_files:
        source = PROJECT_ROOT / relative_path
        if not source.exists():
            missing_files.append(relative_path)

    if missing_files:
        missing_text = '\n'.join(f'- {item}' for item in missing_files)
        raise FileNotFoundError(f'以下打包文件不存在，无法生成云端部署包:\n{missing_text}')


def main():
    output_dir = resolve_project_path(CLOUD_BUNDLE_DIR)
    archive_path = resolve_project_path(CLOUD_BUNDLE_ARCHIVE_PATH)
    validate_bundle_sources(RTDFORMER2_CLOUD_BUNDLE_FILES)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in RTDFORMER2_CLOUD_BUNDLE_FILES:
        source = PROJECT_ROOT / relative_path
        target = output_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    archive_path = Path(
        shutil.make_archive(
            base_name=str(archive_path.with_suffix('')),
            format='zip',
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )
    )

    print(f'云端最小部署目录已生成: {output_dir}')
    print(f'云端最小部署压缩包已生成: {archive_path}')


if __name__ == '__main__':
    main()