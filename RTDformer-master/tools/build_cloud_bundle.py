import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from const import CLOUD_BUNDLE_ARCHIVE_PATH, CLOUD_BUNDLE_DIR, RTDFORMER2_CLOUD_BUNDLE_FILES

def resolve_project_path(project_root, raw_path):
    path = Path(raw_path)
    # 如果是绝对路径直接返回，否则统一基于 project_root
    return path if path.is_absolute() else project_root / path

def main():
    # 统一解析路径
    output_dir = resolve_project_path(PROJECT_ROOT, CLOUD_BUNDLE_DIR)
    archive_path = resolve_project_path(PROJECT_ROOT, CLOUD_BUNDLE_ARCHIVE_PATH)
    
    # 预先验证并获取完整的源文件路径
    sources = [(PROJECT_ROOT / f, output_dir / f) for f in RTDFORMER2_CLOUD_BUNDLE_FILES]
    
    # 验证存在性
    missing = [str(s) for s, t in sources if not s.exists()]
    if missing:
        raise FileNotFoundError(f"缺失文件：{missing}")

    # 清理并创建目录
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 拷贝
    for src, dst in sources:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    
    # 打包（make_archive 会处理覆盖问题）
    shutil.make_archive(
        base_name=str(archive_path.with_suffix('')),
        format='zip',
        root_dir=output_dir.parent,
        base_dir=output_dir.name
    )
    print(f"打包完成：{archive_path}")

if __name__ == '__main__':
    main()