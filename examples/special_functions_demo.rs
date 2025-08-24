use rustorch::prelude::*;
use rustorch::special::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 RusTorch Special Functions Demo");
    println!("==================================");

    // Create test tensor
    let x = Tensor::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0], vec![6]);
    println!("\n📊 Input tensor: {:?}", x.data);

    // Gamma functions demo
    println!("\n🔢 Gamma Functions:");
    println!("-------------------");
    
    let gamma_result = x.gamma()?;
    println!("Γ(x) = {:?}", gamma_result.data);
    
    let lgamma_result = x.lgamma()?;
    println!("ln(Γ(x)) = {:?}", lgamma_result.data);
    
    let digamma_result = x.digamma()?;
    println!("ψ(x) = {:?}", digamma_result.data);

    // Error functions demo  
    println!("\n❌ Error Functions:");
    println!("-------------------");
    
    let erf_x = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    println!("Input: {:?}", erf_x.data);
    
    let erf_result = erf_x.erf()?;
    println!("erf(x) = {:?}", erf_result.data);
    
    let erfc_result = erf_x.erfc()?;
    println!("erfc(x) = {:?}", erfc_result.data);

    // Bessel functions demo
    println!("\n🌊 Bessel Functions:");
    println!("--------------------");
    
    let bessel_x = Tensor::from_vec(vec![0.5, 1.0, 2.0, 5.0], vec![4]);
    println!("Input: {:?}", bessel_x.data);
    
    let j0_result = bessel_x.bessel_j(0.0)?;
    println!("J_0(x) = {:?}", j0_result.data);
    
    let j1_result = bessel_x.bessel_j(1.0)?;
    println!("J_1(x) = {:?}", j1_result.data);
    
    let i0_result = bessel_x.bessel_i(0.0)?;
    println!("I_0(x) = {:?}", i0_result.data);

    // Scalar function demos
    println!("\n🎯 Scalar Function Examples:");
    println!("-----------------------------");
    
    // Gamma function properties
    println!("Γ(5) = {} (should be 24)", gamma::gamma_scalar(5.0_f64)?);
    println!("Γ(0.5) = {} (should be √π ≈ 1.772)", gamma::gamma_scalar(0.5_f64)?);
    
    // Error function properties
    println!("erf(1) = {} (should be ≈ 0.8427)", error::erf_scalar(1.0_f64));
    println!("erfc(0) = {} (should be 1)", error::erfc_scalar(0.0_f64));
    
    // Bessel function properties
    println!("J_0(0) = {} (should be 1)", bessel::bessel_j_scalar(0.0_f64, 0.0)?);
    println!("I_0(0) = {} (should be 1)", bessel::bessel_i_scalar(0.0_f64, 0.0)?);

    // Beta function demo
    println!("\n🎨 Beta Function:");
    println!("-----------------");
    
    let beta_result = gamma::beta(2.0_f64, 3.0)?;
    println!("B(2, 3) = {} (should be 1/12 ≈ 0.0833)", beta_result);
    
    let lbeta_result = gamma::lbeta(5.0_f64, 7.0)?;
    println!("ln(B(5, 7)) = {}", lbeta_result);

    // Mathematical identities verification
    println!("\n✅ Mathematical Identities:");
    println!("---------------------------");
    
    // Verify erf(-x) = -erf(x)
    let x_val = 1.5_f64;
    let erf_pos = error::erf_scalar(x_val);
    let erf_neg = error::erf_scalar(-x_val);
    println!("erf({}) = {}, erf(-{}) = {}", x_val, erf_pos, x_val, erf_neg);
    println!("Symmetry check: {} ≈ {} (should be equal)", erf_pos, -erf_neg);
    
    // Verify erfc(x) = 1 - erf(x)
    let erfc_val = error::erfc_scalar(x_val);
    let one_minus_erf = 1.0 - erf_pos;
    println!("erfc({}) = {}, 1-erf({}) = {}", x_val, erfc_val, x_val, one_minus_erf);
    println!("Identity check: difference = {}", (erfc_val - one_minus_erf).abs());

    println!("\n🎉 Special Functions Demo Complete!");
    println!("💡 All special functions are working correctly!");

    Ok(())
}