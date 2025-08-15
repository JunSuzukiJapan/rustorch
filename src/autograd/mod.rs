use crate::tensor::Tensor;
use std::sync::{Arc, RwLock};
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg};
use ndarray::ArrayD;
use num_traits::Float;
use serde::{Serialize, Deserialize};
use crate::tensor::Tensor;
use std::any::Any;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Variable<T: Float> {
    data: Arc<RwLock<Tensor<T>>>,
    grad: Arc<RwLock<Option<Tensor<T>>>>,
    requires_grad: bool,
    #[serde(skip_serializing, skip_deserializing)]
    grad_fn: Option<Arc<dyn Fn(&Tensor<T>) -> (Option<Tensor<T>>, Option<Tensor<T>>, Option<Tensor<T>>) + Send + Sync>>,
    #[serde(skip_serializing, skip_deserializing)]
    input: Option<Arc<RwLock<Variable<T>>>>,
    #[serde(skip_serializing, skip_deserializing)]
    weight: Option<Arc<RwLock<Variable<T>>>>,
    #[serde(skip_serializing, skip_deserializing)]
    bias: Option<Arc<RwLock<Variable<T>>>>,
    _marker: PhantomData<T>,
}

impl<T: Float + 'static> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Variable {
            data: self.data.clone(),
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
            input: self.input.clone(),
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: Float + 'static> Variable<T> {
    /// Creates a new variable with the given tensor.
    /// 与えられたテンソルで新しい変数を作成します。
    pub fn new(data: Tensor<T>, requires_grad: bool) -> Self {
        Variable {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(if requires_grad {
                Some(Tensor::zeros(data.shape()))
            } else {
                None
            })),
            requires_grad,
            grad_fn: None,
            input: None,
            weight: None,
            bias: None,
            _marker: PhantomData,
        }
    }

    pub fn backward(&mut self, grad_output: Option<Tensor<T>>, retain_graph: bool) {
        if !self.requires_grad {
            println!("Variable does not require gradient, skipping backward pass");
            return;
        }

        // Initialize gradient if needed
        let grad_output = grad_output.unwrap_or_else(|| {
            println!("Initializing gradient with ones");
            Tensor::ones(&[])
        });

        // Set or accumulate the gradient
        if let Err(e) = self.add_grad(grad_output) {
            println!("Error adding gradient: {}", e);
            return;
        }

        // If we have a gradient function, use it to compute gradients
        if let Some(ref grad_fn) = self.grad_fn {
            println!("Computing gradients with grad_fn");
            let (grad_input, grad_weight, grad_bias) = grad_fn(self.grad.as_ref().unwrap());
            
            // Propagate gradients to input if it exists and requires gradients
            if let Some(ref mut input_var) = self.input {
                if let Some(grad) = grad_input {
                    if input_var.read().unwrap().requires_grad {
                        println!("Propagating gradient to input: {:?}", grad.shape());
                        input_var.write().unwrap().backward(Some(grad), retain_graph);
                    }
                }
            }
            
            // Handle weight gradient if it exists
            if let Some(ref mut weight_var) = self.weight {
                if let Some(grad) = grad_weight {
                    if weight_var.read().unwrap().requires_grad {
                        println!("Setting weight gradient: {:?}", grad.shape());
                        if let Err(e) = weight_var.write().unwrap().add_grad(grad) {
                            println!("Error adding weight gradient: {}", e);
                        }
                    }
                }
            }
            
            // Handle bias gradient if it exists
            if let Some(ref mut bias_var) = self.bias {
                if let Some(grad) = grad_bias {
                    if bias_var.read().unwrap().requires_grad {
                        println!("Setting bias gradient: {:?}", grad.shape());
                        if let Err(e) = bias_var.write().unwrap().add_grad(grad) {
                            println!("Error adding bias gradient: {}", e);
                        }
                    }
                }
            }
        }
        
        // Clean up if we're not retaining the graph
        if !retain_graph {
            self.input = None;
            self.weight = None;
            self.bias = None;
            self.grad_fn = None;
        }
    }

    pub fn add_grad(&mut self, grad: Tensor<T>) -> Result<(), String> {
        if !self.requires_grad {
            return Err("Cannot add gradient to a variable that does not require gradient".to_string());
        }

        match &mut self.grad {
            Some(existing_grad) => {
                if existing_grad.shape() != grad.shape() {
                    return Err(format!("Shape mismatch: expected {:?}, got {:?}", 
                                      existing_grad.shape(), grad.shape()));
                }
                *existing_grad = existing_grad.clone() + &grad;
            },
            None => {
                self.grad = Some(grad);
            }
        }
        Ok(())
    }

    pub fn set_grad_fn(&mut self, 
                      f: Box<dyn Fn(&Tensor<T>) -> (Option<Tensor<T>>, Option<Tensor<T>>, Option<Tensor<T>>) + Send + Sync>) {
        self.grad_fn = Some(f);
    }

    pub fn set_input(&mut self, input: Variable<T>) {
        self.input = Some(Arc::new(RwLock::new(input)));
    }

    pub fn set_weight(&mut self, weight: Variable<T>) {
        self.weight = Some(Arc::new(RwLock::new(weight)));
    }

    pub fn set_bias(&mut self, bias: Option<Variable<T>>) {
        self.bias = bias.map(|b| Arc::new(RwLock::new(b)));
    }

    pub fn data(&self) -> &Tensor<T> {
        &self.data
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn grad(&self) -> Option<&Tensor<T>> {
        self.grad.as_ref()
    }
}
