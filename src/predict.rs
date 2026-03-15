use candle_core::{Result, Tensor};
use candle_nn::ops::softmax;
use crate::model::MLP;

pub fn predict(model:&MLP, image: &Tensor,true_label:u8 )-> Result<()>{

    let image = image.unsqueeze(0)?; //normal img coming 784 need to add dimension[1,784]
    let logits= model.forward(&image)?;

    let probs =softmax(&logits,1)?; //softmax turn raw score into prabability to1.0

    let probs = probs.squeeze(0)?;

    let predicted = probs.argmax(0)?.to_scalar::<u32>()?;

    println!("\n--- Prediction ---");
    for digit in 0..10u32 {
        let p = probs.get(digit as usize)?.to_scalar::<f32>()?;
        let bar = "█".repeat((p * 30.0) as usize);
        println!("  {} | {:5.1}% {}", digit, p * 100.0, bar);
    }
    println!("\n  Predicted digit: {}", predicted);

    if predicted == true_label as u32{
        println!("Correct");

    }   else{
        println!("Wrong True label is {}",true_label);
    }

    Ok(())
}