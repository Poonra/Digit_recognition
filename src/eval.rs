use candle_core::{DType, Result};
use crate::model::MLP;
use crate::data::MnistData;

pub fn evaluate(mnist: &MnistData,model: &MLP)->Result<()>{
    let test_images = &mnist.test_images;
    let test_labels = &mnist.test_labels;

    let logits = model.forward(test_images)?;

    //argmax
    let predictions = logits.argmax(1)?;//argmax predict the highest index
//cast both to same type so can compare
    let predictions = predictions.to_dtype(DType::U32)?;
    let labels = test_labels.to_dtype(DType::U32)?;

    let correct = predictions.eq(&labels)?; //1 when correct 0 when wrong
    
    //sum of all the 1s / total
    let correct_count =  correct.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    let accuracy = correct_count / 10000.0 * 100.0;

    println!("Test accuracy {:.2}", accuracy);

    Ok(())
}