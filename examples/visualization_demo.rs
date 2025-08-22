//! 可視化機能のデモンストレーション
//! Visualization functionality demonstration

use rustorch::prelude::*;
use rustorch::visualization::*;
use rustorch::models::high_level::TrainingHistory;
use rustorch::visualization::utils::{ColorPalette, create_dashboard};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 RusTorch 可視化機能デモ");
    println!("🎨 RusTorch Visualization Demo\n");

    // ===== 1. 学習履歴の可視化 =====
    println!("📈 1. 学習曲線の可視化 / Training Curves Visualization");
    
    let mut history = TrainingHistory::<f32>::new();
    
    // ダミーの学習履歴を生成
    for epoch in 1..=10 {
        let train_loss = 1.0 - (epoch as f32 * 0.08);
        let val_loss = train_loss + 0.05;
        
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), vec![(epoch as f64 * 0.07 + 0.3).min(0.95)]);
        metrics.insert("precision".to_string(), vec![(epoch as f64 * 0.06 + 0.4).min(0.92)]);
        
        history.train_loss.push(train_loss);
        history.val_loss.push(val_loss);
        for (key, value) in metrics {
            history.metrics.entry(key).or_insert_with(Vec::new).extend(value);
        }
    }
    
    // プロッターを作成
    let plotter = TrainingPlotter::with_config(PlotConfig {
        width: 800,
        height: 600,
        dpi: 150,
        chart_type: ChartType::Single,
        background_color: "#ffffff".to_string(),
        font_size: 14,
        line_width: 2.5,
        marker_size: 5.0,
    });
    
    // 学習曲線をプロット
    match plotter.plot_training_curves(&history) {
        Ok(svg) => {
            println!("✓ 学習曲線SVG生成成功 ({} bytes)", svg.len());
            
            // ファイルに保存
            if let Err(e) = plotter.save_plot(&svg, "training_curves.svg") {
                println!("⚠ ファイル保存に失敗: {}", e);
            } else {
                println!("✓ ファイルに保存: training_curves.svg");
            }
        }
        Err(e) => println!("✗ 学習曲線生成エラー: {}", e),
    }
    
    // メトリクス時系列
    match plotter.plot_metrics_timeline(&history, "accuracy") {
        Ok(svg) => {
            println!("✓ 精度時系列SVG生成成功 ({} bytes)", svg.len());
        }
        Err(e) => println!("⚠ 精度時系列生成エラー: {}", e),
    }
    
    println!();

    // ===== 2. テンソルの可視化 =====
    println!("🔢 2. テンソルの可視化 / Tensor Visualization");
    
    // テンソルビジュアライザーを作成
    let tensor_viz = TensorVisualizer::with_config(TensorPlotConfig {
        colormap: ColorMap::Viridis,
        normalize: true,
        aspect: "equal".to_string(),
        title: Some("Sample Heatmap".to_string()),
        show_colorbar: true,
        show_values: false,
        precision: 3,
    });
    
    // 2Dテンソルのヒートマップ
    let heat_data: Vec<f32> = (0..16).map(|i| (i as f32 / 4.0).sin()).collect();
    let heat_tensor = Tensor::from_vec(heat_data, vec![4, 4]);
    
    match tensor_viz.plot_heatmap(&heat_tensor) {
        Ok(svg) => {
            println!("✓ ヒートマップSVG生成成功 ({} bytes)", svg.len());
            
            // ファイルに保存
            if let Err(e) = save_plot(&svg, "heatmap.svg", PlotFormat::Svg) {
                println!("⚠ ファイル保存に失敗: {}", e);
            } else {
                println!("✓ ファイルに保存: heatmap.svg");
            }
        }
        Err(e) => println!("✗ ヒートマップ生成エラー: {}", e),
    }
    
    // 1Dテンソルの棒グラフ
    let bar_data = vec![1.0, 3.0, 2.0, 4.0, 1.5, 3.5, 2.5];
    let bar_tensor = Tensor::from_vec(bar_data, vec![7]);
    
    match tensor_viz.plot_bar_chart(&bar_tensor) {
        Ok(svg) => {
            println!("✓ 棒グラフSVG生成成功 ({} bytes)", svg.len());
        }
        Err(e) => println!("✗ 棒グラフ生成エラー: {}", e),
    }
    
    // 3Dテンソルのスライス
    let slice_data: Vec<f32> = (0..24).map(|i| (i as f32).cos()).collect();
    let slice_tensor = Tensor::from_vec(slice_data, vec![2, 3, 4]);
    
    match tensor_viz.plot_3d_slices(&slice_tensor) {
        Ok(svg) => {
            println!("✓ 3Dスライス可視化SVG生成成功 ({} bytes)", svg.len());
        }
        Err(e) => println!("✗ 3Dスライス生成エラー: {}", e),
    }
    
    println!();

    // ===== 3. 計算グラフの可視化 =====
    println!("🕸️ 3. 計算グラフの可視化 / Computation Graph Visualization");
    
    // 計算グラフビジュアライザーを作成
    let mut graph_viz = GraphVisualizer::with_layout(GraphLayout::Hierarchical);
    
    // テスト用変数を作成
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let variable = Variable::new(tensor, true);
    
    // グラフを構築
    match graph_viz.build_graph(&variable) {
        Ok(()) => {
            println!("✓ 計算グラフ構築成功");
            
            // SVG形式で出力
            match graph_viz.to_svg() {
                Ok(svg) => {
                    println!("✓ 計算グラフSVG生成成功 ({} bytes)", svg.len());
                    
                    // ファイルに保存
                    if let Err(e) = save_plot(&svg, "computation_graph.svg", PlotFormat::Svg) {
                        println!("⚠ ファイル保存に失敗: {}", e);
                    } else {
                        println!("✓ ファイルに保存: computation_graph.svg");
                    }
                }
                Err(e) => println!("✗ SVG生成エラー: {}", e),
            }
            
            // DOT形式で出力
            match graph_viz.to_dot() {
                Ok(dot) => {
                    println!("✓ DOT形式生成成功 ({} bytes)", dot.len());
                    if let Err(e) = save_plot(&dot, "computation_graph.dot", PlotFormat::Dot) {
                        println!("⚠ DOTファイル保存に失敗: {}", e);
                    } else {
                        println!("✓ DOTファイルに保存: computation_graph.dot");
                    }
                }
                Err(e) => println!("✗ DOT生成エラー: {}", e),
            }
            
            // グラフ統計情報
            let stats = graph_viz.get_statistics();
            println!("📊 グラフ統計:");
            for (key, value) in stats {
                println!("   {}: {}", key, value);
            }
        }
        Err(e) => println!("✗ 計算グラフ構築エラー: {}", e),
    }
    
    println!();

    // ===== 4. カラーパレットのデモ =====
    println!("🎨 4. カラーパレット / Color Palette Demo");
    
    let categorical_colors = ColorPalette::categorical();
    println!("✓ カテゴリカル色数: {}", categorical_colors.len());
    
    for i in 0..3 {
        let color = ColorPalette::get_categorical_color(i);
        println!("   色 {}: {}", i, color);
    }
    
    let sequential_colors = ColorPalette::sequential_blues();
    println!("✓ シーケンシャル色数: {}", sequential_colors.len());
    
    for &value in &[0.0, 0.5, 1.0] {
        let color = ColorPalette::get_sequential_color(value);
        println!("   値 {}: {}", value, color);
    }
    
    println!();

    // ===== 5. ダッシュボード作成 =====
    println!("📊 5. ダッシュボード作成 / Dashboard Creation");
    
    // サンプルプロット
    let sample_plots = vec![
        ("Training Loss", "<svg width=\"300\" height=\"200\"><rect x=\"10\" y=\"10\" width=\"280\" height=\"180\" fill=\"#e3f2fd\" stroke=\"#1976d2\"/><text x=\"150\" y=\"100\" text-anchor=\"middle\" font-family=\"Arial\">Training Loss Chart</text></svg>"),
        ("Validation Accuracy", "<svg width=\"300\" height=\"200\"><rect x=\"10\" y=\"10\" width=\"280\" height=\"180\" fill=\"#e8f5e8\" stroke=\"#4caf50\"/><text x=\"150\" y=\"100\" text-anchor=\"middle\" font-family=\"Arial\">Accuracy Chart</text></svg>"),
        ("Model Architecture", "<svg width=\"300\" height=\"200\"><rect x=\"10\" y=\"10\" width=\"280\" height=\"180\" fill=\"#fff3e0\" stroke=\"#ff9800\"/><text x=\"150\" y=\"100\" text-anchor=\"middle\" font-family=\"Arial\">Model Diagram</text></svg>"),
    ];
    
    match create_dashboard(sample_plots) {
        Ok(dashboard_html) => {
            println!("✓ ダッシュボードHTML生成成功 ({} bytes)", dashboard_html.len());
            
            // HTMLファイルに保存
            if let Err(e) = save_plot(&dashboard_html, "dashboard.html", PlotFormat::Html) {
                println!("⚠ ダッシュボード保存に失敗: {}", e);
            } else {
                println!("✓ ダッシュボード保存: dashboard.html");
                println!("  ブラウザで開いて表示可能です");
            }
        }
        Err(e) => println!("✗ ダッシュボード生成エラー: {}", e),
    }
    
    println!();

    // ===== まとめ =====
    println!("🎉 可視化機能デモ完了！");
    println!("🎉 Visualization Demo Complete!\n");
    
    println!("生成されたファイル:");
    println!("Generated files:");
    println!("  - training_curves.svg     : 学習曲線");
    println!("  - heatmap.svg             : テンソルヒートマップ");
    println!("  - computation_graph.svg   : 計算グラフ (SVG)");
    println!("  - computation_graph.dot   : 計算グラフ (DOT)");
    println!("  - dashboard.html          : 可視化ダッシュボード");
    println!();
    
    println!("💡 使用方法:");
    println!("💡 Usage:");
    println!("  - SVGファイルはブラウザやベクター画像エディタで表示");
    println!("  - DOTファイルはGraphvizで変換可能 (dot -Tpng file.dot -o output.png)");
    println!("  - HTMLダッシュボードはブラウザで直接表示");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_visualization_demo_components() {
        // 基本的なコンポーネントのテスト
        
        // TrainingPlotter
        let plotter = TrainingPlotter::new();
        assert_eq!(plotter.config.width, 800);
        
        // TensorVisualizer  
        let visualizer = TensorVisualizer::new();
        assert_eq!(visualizer.config.colormap, ColorMap::Viridis);
        
        // GraphVisualizer
        let graph_viz = GraphVisualizer::new();
        assert_eq!(graph_viz.layout, GraphLayout::Hierarchical);
        
        // ColorPalette
        let colors = ColorPalette::categorical();
        assert!(!colors.is_empty());
        
        println!("✓ All visualization components initialized successfully");
    }
    
    #[test]
    fn test_tensor_creation_and_visualization() {
        // テンソル作成と可視化のテスト
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), vec![2, 2]);
        
        // 形状確認
        assert_eq!(tensor.shape(), &vec![2, 2]);
        
        // データ確認
        if let Some(slice) = tensor.as_slice() {
            assert_eq!(slice, &data[..]);
        }
        
        // 可視化テスト
        let visualizer = TensorVisualizer::new();
        let result = visualizer.plot_heatmap(&tensor);
        assert!(result.is_ok(), "Tensor visualization should succeed");
        
        println!("✓ Tensor creation and visualization test passed");
    }
}