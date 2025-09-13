window.BENCHMARK_DATA = {
  "lastUpdate": 1757746545711,
  "repoUrl": "https://github.com/JunSuzukiJapan/rustorch",
  "entries": {
    "RusTorch Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "committer": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "distinct": true,
          "id": "157121ff0cf581435511c23d7aa59f14bb3dd213",
          "message": "fix: resolve gh-pages branch conflicts in CI workflows\n\n- Add dependency between docs and benchmark jobs to prevent concurrent writes\n- Change docs deployment to preserve existing files instead of force orphan\n- Prevent git push conflicts when multiple jobs write to gh-pages simultaneously\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2025-09-10T14:35:05+09:00",
          "tree_id": "ae0cc387ebb6ef551144575eef9b83acb8873163",
          "url": "https://github.com/JunSuzukiJapan/rustorch/commit/157121ff0cf581435511c23d7aa59f14bb3dd213"
        },
        "date": 1757482570198,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "RusTorch Quick Benchmark",
            "value": 30041,
            "range": "±5%",
            "unit": "ms",
            "extra": "Rust tensor operations benchmark"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "committer": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "distinct": true,
          "id": "b5ad8d12eead070bea8cdc933dd9f716a58432cd",
          "message": "fix: add missing BLAS/LAPACK dependencies in nightly workflow\n\n- Install libopenblas-dev, liblapack-dev, libblas-dev for linalg-system feature\n- Add library verification and symlink creation\n- Set proper PKG_CONFIG_PATH and LD_LIBRARY_PATH environment variables\n- Fix 'cannot find -llapack: No such file or directory' linker error\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2025-09-10T18:53:14+09:00",
          "tree_id": "03fd6c0a1848276acffb88ade9f5838d9c71a398",
          "url": "https://github.com/JunSuzukiJapan/rustorch/commit/b5ad8d12eead070bea8cdc933dd9f716a58432cd"
        },
        "date": 1757498060401,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "RusTorch Quick Benchmark",
            "value": 30011,
            "range": "±5%",
            "unit": "ms",
            "extra": "Rust tensor operations benchmark"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "committer": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "distinct": true,
          "id": "c111b18f14d07a92c7083833e3bb0e1a621d2665",
          "message": "fix: remove unsupported --output-format option from cargo bench\n\n- Replace --output-format json with tee to capture benchmark output\n- Fix 'Unrecognized option: output-format' error in nightly benchmarks\n- Maintain benchmark results collection for performance regression analysis\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2025-09-11T19:09:40+09:00",
          "tree_id": "6636a38d7cc221fde489709a4821dbd66378930b",
          "url": "https://github.com/JunSuzukiJapan/rustorch/commit/c111b18f14d07a92c7083833e3bb0e1a621d2665"
        },
        "date": 1757585449560,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "RusTorch Quick Benchmark",
            "value": 30010,
            "range": "±5%",
            "unit": "ms",
            "extra": "Rust tensor operations benchmark"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "committer": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "distinct": true,
          "id": "6393b7b6c8581fbc67e8b69bc6a4f0a9a4943dcf",
          "message": "fix: resolve Windows heap corruption in nightly tests\n\n- Use --no-default-features for Windows to avoid BLAS/LAPACK linking issues\n- Skip stress tests on Windows to prevent STATUS_HEAP_CORRUPTION (0xc0000374)\n- Maintain full feature testing on Linux and macOS where libraries work properly\n- Improve Windows CI stability while preserving comprehensive testing on other platforms\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2025-09-12T13:27:27+09:00",
          "tree_id": "fc09e5434a230ab3e2f12a1b6bd977309aadee97",
          "url": "https://github.com/JunSuzukiJapan/rustorch/commit/6393b7b6c8581fbc67e8b69bc6a4f0a9a4943dcf"
        },
        "date": 1757651309877,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "RusTorch Quick Benchmark",
            "value": 30012,
            "range": "±5%",
            "unit": "ms",
            "extra": "Rust tensor operations benchmark"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "committer": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "distinct": true,
          "id": "50342e457a65bf7a26483ade5415950e82c0375e",
          "message": "update: local settings and final cleanup\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2025-09-12T13:29:03+09:00",
          "tree_id": "742fc1115ed95fe04382975604598fe4c895feb0",
          "url": "https://github.com/JunSuzukiJapan/rustorch/commit/50342e457a65bf7a26483ade5415950e82c0375e"
        },
        "date": 1757651398266,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "RusTorch Quick Benchmark",
            "value": 30010,
            "range": "±5%",
            "unit": "ms",
            "extra": "Rust tensor operations benchmark"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "committer": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "distinct": true,
          "id": "f1fe7ab0a67e9f865c0e2b5d6bec19661683f7b8",
          "message": "fix: implement conservative Windows testing strategy for nightly CI\n\n- Skip problematic auto_device and GPU-related tests on Windows\n- Use single-threaded execution to prevent heap corruption\n- Focus on core functionality testing only on Windows platform\n- Addresses STATUS_HEAP_CORRUPTION issues in nightly Windows tests\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2025-09-13T14:34:29+09:00",
          "tree_id": "a472fa0081a8a88f2cfdb1d04277dcd634720143",
          "url": "https://github.com/JunSuzukiJapan/rustorch/commit/f1fe7ab0a67e9f865c0e2b5d6bec19661683f7b8"
        },
        "date": 1757741756169,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "RusTorch Quick Benchmark",
            "value": 30008,
            "range": "±5%",
            "unit": "ms",
            "extra": "Rust tensor operations benchmark"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jun.suzuki.japan@gmail.com",
            "name": "JunSuzukiJapan",
            "username": "JunSuzukiJapan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2365939a9d83ed433a38eb9b80f5b8b99ad942d2",
          "message": "feat: synchronize all version references to 0.6.18 (#26)\n\n- Updated Cargo.toml from 0.6.17 to 0.6.18\n- Updated all Jupyter notebooks in notebooks/ to use rustorch 0.6.18\n- Updated jupyter/package.json to version 0.6.18\n- Synchronized multilingual notebooks: en, es, fr, it, ko, zh, ja\n- Ensured consistent version alignment across all components\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2025-09-13T15:54:47+09:00",
          "tree_id": "d701e1f7a58bd8335e3b1b6a7aaa5ace37a7a3f0",
          "url": "https://github.com/JunSuzukiJapan/rustorch/commit/2365939a9d83ed433a38eb9b80f5b8b99ad942d2"
        },
        "date": 1757746544621,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "RusTorch Quick Benchmark",
            "value": 30010,
            "range": "±5%",
            "unit": "ms",
            "extra": "Rust tensor operations benchmark"
          }
        ]
      }
    ]
  }
}