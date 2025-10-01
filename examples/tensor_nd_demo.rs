use rustorch::tensor_nd;

fn main() {
    println!("=== RusTorch tensor_nd! Macro Demo ===\n");

    // 1D tensor
    println!("1D Tensor:");
    let t1 = tensor_nd!([1, 2, 3, 4, 5]);
    println!("  Shape: {:?}", t1.shape());
    println!("  Data: {:?}\n", t1.data);

    // 2D tensor
    println!("2D Tensor:");
    let t2 = tensor_nd!([[1, 2, 3], [4, 5, 6]]);
    println!("  Shape: {:?}", t2.shape());
    println!("  Data: {:?}\n", t2.data);

    // 3D tensor
    println!("3D Tensor:");
    let t3 = tensor_nd!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    println!("  Shape: {:?}", t3.shape());
    println!("  Data: {:?}\n", t3.data);

    // 4D tensor (NEW!)
    println!("4D Tensor:");
    let t4 = tensor_nd!([
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
    ]);
    println!("  Shape: {:?}", t4.shape());
    println!("  Data: {:?}\n", t4.data);

    // 5D tensor (NEW!)
    println!("5D Tensor:");
    let t5 = tensor_nd!([[[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]],]);
    println!("  Shape: {:?}", t5.shape());
    println!("  Data: {:?}\n", t5.data);

    // 6D tensor (NEW!)
    println!("6D Tensor:");
    let t6 = tensor_nd!([[[[[[1, 2]]]]]]);
    println!("  Shape: {:?}", t6.shape());
    println!("  Data: {:?}\n", t6.data);

    // Mixed types
    println!("Mixed Types (integers and floats):");
    let tm = tensor_nd!([1, 2.5, 3, 4.7, 5]);
    println!("  Shape: {:?}", tm.shape());
    println!("  Data: {:?}\n", tm.data);

    // Realistic 4D example (batch_size, channels, height, width)
    println!("Realistic 4D Tensor (batch_size=2, channels=3, height=4, width=4):");
    let realistic = tensor_nd!([
        // Batch 1
        [
            // Channel 1 (Red)
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ],
            // Channel 2 (Green)
            [
                [17, 18, 19, 20],
                [21, 22, 23, 24],
                [25, 26, 27, 28],
                [29, 30, 31, 32]
            ],
            // Channel 3 (Blue)
            [
                [33, 34, 35, 36],
                [37, 38, 39, 40],
                [41, 42, 43, 44],
                [45, 46, 47, 48]
            ]
        ],
        // Batch 2
        [
            // Channel 1 (Red)
            [
                [49, 50, 51, 52],
                [53, 54, 55, 56],
                [57, 58, 59, 60],
                [61, 62, 63, 64]
            ],
            // Channel 2 (Green)
            [
                [65, 66, 67, 68],
                [69, 70, 71, 72],
                [73, 74, 75, 76],
                [77, 78, 79, 80]
            ],
            // Channel 3 (Blue)
            [
                [81, 82, 83, 84],
                [85, 86, 87, 88],
                [89, 90, 91, 92],
                [93, 94, 95, 96]
            ]
        ]
    ]);
    println!("  Shape: {:?}", realistic.shape());
    println!(
        "  First few values: {:?}...\n",
        &realistic.data.as_slice().unwrap()[0..5]
    );

    println!("=== All tensor_nd! macro demonstrations completed successfully! ===");
}
