#!/usr/bin/env python3
"""
Python環境診断スクリプト - PyO3ビルドの問題を特定
"""

import sys
import sysconfig
import os
import subprocess
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    print(f"\n📋 {title}")
    print("-" * 40)

def check_python_info():
    print_section("Python基本情報")
    print(f"Python実行ファイル: {sys.executable}")
    print(f"Pythonバージョン: {sys.version}")
    print(f"プラットフォーム: {sys.platform}")
    
def check_python_config():
    print_section("Python設定情報")
    
    # Framework check
    framework = sysconfig.get_config_var('PYTHONFRAMEWORK')
    print(f"PYTHONFRAMEWORK: {framework}")
    
    if framework == 'Python':
        print("⚠️  Framework Python検出!")
        print("   macOSのFramework版Python (Homebrew/Python.org) を使用しています")
        print("   PyO3では特別な設定が必要です")
    else:
        print("✅ 通常のPython (Framework以外)")
    
    # Other important config vars
    config_vars = [
        'LIBDIR', 'INCLUDEDIR', 'LDLIBRARY', 
        'LIBRARY', 'LDSHARED', 'CC'
    ]
    
    for var in config_vars:
        value = sysconfig.get_config_var(var)
        print(f"{var}: {value}")

def check_cargo_config():
    print_section("Cargo設定チェック")
    
    config_path = Path(".cargo/config.toml")
    
    if config_path.exists():
        print(f"✅ .cargo/config.toml が存在します")
        
        try:
            content = config_path.read_text()
            print("\n設定内容:")
            print(content)
            
            if "dynamic_lookup" in content:
                print("✅ dynamic_lookup設定が見つかりました")
            else:
                print("⚠️  dynamic_lookup設定が見つかりません")
                
        except Exception as e:
            print(f"❌ 設定ファイル読み込みエラー: {e}")
    else:
        print("❌ .cargo/config.toml が見つかりません")

def check_build_requirements():
    print_section("ビルド要件チェック")
    
    # Check Rust
    try:
        result = subprocess.run(["cargo", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Cargo: {result.stdout.strip()}")
        else:
            print("❌ Cargoが見つかりません")
    except FileNotFoundError:
        print("❌ Cargoがインストールされていません")
    
    # Check Python dev headers
    include_dir = sysconfig.get_config_var('INCLUDEDIR')
    if include_dir and os.path.exists(include_dir):
        print(f"✅ Python開発用ヘッダー: {include_dir}")
    else:
        print("⚠️  Python開発用ヘッダーが見つかりません")

def provide_recommendations():
    print_section("推奨設定")
    
    framework = sysconfig.get_config_var('PYTHONFRAMEWORK')
    config_path = Path(".cargo/config.toml")
    
    if framework == 'Python' and not config_path.exists():
        print("🔧 Framework Python用の設定が必要です:")
        print()
        print("以下の内容で .cargo/config.toml を作成してください:")
        print()
        print("[build]")
        print("rustflags = [")
        print('    "-C", "link-arg=-undefined",')
        print('    "-C", "link-arg=dynamic_lookup",')
        print("]")
        print()
        print("[env]")
        print(f'PYO3_PYTHON = "{sys.executable}"')
        print()
        print("その後:")
        print("cargo clean && cargo build --release")
        
    elif framework == 'Python':
        print("✅ Framework Python用の設定済み")
        
    else:
        print("✅ 特別な設定は不要です")

def check_alternative_pythons():
    print_section("代替Python環境")
    
    alternatives = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3",
    ]
    
    print("利用可能なPython実行ファイル:")
    for alt in alternatives:
        if os.path.exists(alt):
            try:
                result = subprocess.run([alt, "--version"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {alt}: {result.stdout.strip()}")
            except:
                print(f"❌ {alt}: 実行エラー")
        else:
            print(f"❌ {alt}: 見つかりません")

def main():
    print_header("RusTorch PyO3 環境診断")
    
    check_python_info()
    check_python_config()
    check_cargo_config()
    check_build_requirements()
    provide_recommendations()
    check_alternative_pythons()
    
    print_header("診断完了")
    print("詳細な解決方法は README.md を参照してください")

if __name__ == "__main__":
    main()