#!/usr/bin/env python3
"""
PyO3ビルド用の自動設定スクリプト
"""

import sys
import sysconfig
import os
from pathlib import Path

def create_cargo_config():
    """Framework Python用のCargo設定を自動作成"""
    
    framework = sysconfig.get_config_var('PYTHONFRAMEWORK')
    
    if framework != 'Python':
        print("✅ Framework Python以外なので、特別な設定は不要です")
        return
    
    print("🔧 Framework Python検出 - Cargo設定を作成中...")
    
    # .cargoディレクトリを作成
    cargo_dir = Path(".cargo")
    cargo_dir.mkdir(exist_ok=True)
    
    config_path = cargo_dir / "config.toml"
    
    # 設定内容
    config_content = f"""[build]
rustflags = [
    "-C", "link-arg=-undefined",
    "-C", "link-arg=dynamic_lookup",
]

[env]
PYO3_PYTHON = "{sys.executable}"
"""
    
    # 既存の設定をバックアップ
    if config_path.exists():
        backup_path = config_path.with_suffix('.toml.backup')
        print(f"📦 既存の設定を {backup_path} にバックアップ")
        config_path.rename(backup_path)
    
    # 新しい設定を書き込み
    config_path.write_text(config_content)
    
    print(f"✅ {config_path} を作成しました")
    print("\n設定内容:")
    print(config_content)
    
    print("🚀 次のステップ:")
    print("   cargo clean && cargo build --release")
    print("   cp target/release/lib_rustorch_py.dylib _rustorch_py.so")

def main():
    print("🐍 RusTorch PyO3 ビルド設定")
    print("=" * 50)
    
    print(f"Python: {sys.executable}")
    print(f"バージョン: {sys.version.split()[0]}")
    
    framework = sysconfig.get_config_var('PYTHONFRAMEWORK')
    print(f"Framework: {framework or 'なし'}")
    
    print()
    create_cargo_config()

if __name__ == "__main__":
    main()