use rustorch::tensor_nd;

#[test]
fn test_tensor_nd_4d() {
    let t = tensor_nd!([
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
    ]);
    assert_eq!(t.shape(), &[2, 2, 2, 2]);
    assert_eq!(t.data.as_slice().unwrap()[0], 1.0);
    assert_eq!(t.data.as_slice().unwrap()[15], 16.0);
}

#[test]
fn test_tensor_nd_5d() {
    let t = tensor_nd!([[[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]],]);
    assert_eq!(t.shape(), &[1, 2, 2, 1, 2]);
    assert_eq!(t.data.as_slice().unwrap().len(), 8);
}

#[test]
fn test_tensor_nd_6d() {
    let t = tensor_nd!([[[[[[1, 2]]]]]]);
    assert_eq!(t.shape(), &[1, 1, 1, 1, 1, 2]);
    assert_eq!(t.data.as_slice().unwrap(), &[1.0, 2.0]);
}

#[test]
fn test_tensor_nd_7d() {
    let t = tensor_nd!([[[[[[[1, 2]]]]]]]);
    assert_eq!(t.shape(), &[1, 1, 1, 1, 1, 1, 2]);
    assert_eq!(t.data.as_slice().unwrap(), &[1.0, 2.0]);
}

#[test]
fn test_tensor_nd_8d() {
    let t = tensor_nd!([[[[[[[[1]]]]]]]]);
    assert_eq!(t.shape(), &[1, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(t.data.as_slice().unwrap(), &[1.0]);
}

#[test]
fn test_tensor_nd_1d() {
    let t = tensor_nd!([1, 2, 3, 4, 5]);
    assert_eq!(t.shape(), &[5]);
    assert_eq!(t.data.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_tensor_nd_2d() {
    let t = tensor_nd!([[1, 2, 3], [4, 5, 6]]);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.data.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_tensor_nd_3d() {
    let t = tensor_nd!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    assert_eq!(t.shape(), &[2, 2, 2]);
    assert_eq!(
        t.data.as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn test_tensor_nd_large_4d() {
    let t = tensor_nd!([
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
        [
            [[13, 14], [15, 16]],
            [[17, 18], [19, 20]],
            [[21, 22], [23, 24]]
        ]
    ]);
    assert_eq!(t.shape(), &[2, 3, 2, 2]);
    assert_eq!(t.data.as_slice().unwrap().len(), 24);
}

#[test]
fn test_tensor_nd_mixed_types() {
    let t = tensor_nd!([1, 2.5, 3, 4.7, 5]);
    assert_eq!(t.shape(), &[5]);
    assert_eq!(t.data.as_slice().unwrap()[0], 1.0);
    assert_eq!(t.data.as_slice().unwrap()[1], 2.5);
    assert_eq!(t.data.as_slice().unwrap()[3], 4.7);
}
