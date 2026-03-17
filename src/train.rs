use candle_core::{DType, Result,Tensor,Device };
use candle_nn::{loss, Optimizer, VarMap};
use indicatif::{ProgressBar, ProgressStyle};
use crate::model::MLP;
use crate::data::MnistData;
use serde::{Deserialize};
use std::fs;

const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 5;
const BATCH_SIZE: usize = 64;
const FINETUNE_REPEAT: usize = 550; 
#[derive(Deserialize)]
struct Sample {
    pixels: Vec<f32>,
    label: u32,
}

pub fn train(
    mnist: &MnistData,
    varmap:&VarMap,
    model:&MLP,
) -> Result<()> {
    let mut opt = candle_nn::SGD::new(varmap.all_vars(),LEARNING_RATE)?;

    for epoch in 0..EPOCHS{
        let pb = ProgressBar::new(60000/ BATCH_SIZE as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{epoch} [{bar:40}] {pos}/{len} loss ={msg}")
            .unwrap()
            .progress_chars("█░░"));
        let mut total_loss = 0f32;
        let mut batches = 0;

        let n = 60000;
        for i in(0..n).step_by(BATCH_SIZE){
            let end = (i+BATCH_SIZE).min(n);

            let images = mnist.train_images.narrow(0,i,end-i)?;
            let labels = mnist.train_labels.narrow(0,i,end-i)?;

            let logits=model.forward(&images)?;

            //convert label to u32 for loss fn
            let labels_u32= labels.to_dtype(DType::U32)?;

            let batch_loss = loss::cross_entropy(&logits, &labels_u32)?;

            //backpropagation
            opt.backward_step(&batch_loss)?;

            total_loss += batch_loss.to_scalar::<f32>()?;
            batches+=1;
            pb.inc(1);
            pb.set_message(format!("{:.4}", total_loss / batches as f32));
           
        }
         pb.finish_with_message(format!("epoch {} abg loss: {:.4}",epoch+1, total_loss/batches as f32));
    }
    Ok(())
}

pub fn finetune_on_collected(
    varmap: &VarMap,
    model: &MLP,
    device: &Device,

)-> Result<()> {
    //load data 
    let data = match fs::read_to_string("collected_data.json"){
        Ok(d) => d,
        Err(_) =>{
            println!("No collected data found, skipping fine-tune.");
            return Ok(());
        }
    };

    let samples: Vec<Sample> =serde_json::from_str(&data).unwrap_or_default();

    if samples.is_empty(){
        println!("collected data is empty, skipping... ");
        return Ok(());
    }

    println!("Fine-tuning on {} collected samples ({}x repeat)...",
        samples.len(), FINETUNE_REPEAT);

    //build tensor from collected sample + give more weight
    let mut all_pixels: Vec<f32> = Vec::new();
    let mut all_labels: Vec<u32> = Vec::new();

    for _ in 0..FINETUNE_REPEAT {
        for s in &samples{
            all_pixels.extend_from_slice(&s.pixels);
            all_labels.push(s.label);
        }
    }

    let n = all_labels.len();
    let images = Tensor::from_vec(all_pixels, (n, 784), device)?;
    let labels = Tensor::from_vec(all_labels, (n,), device)?;

    let mut opt = candle_nn::SGD::new(varmap.all_vars(), 0.05)?; // overwrite the mnist!!!

    let pb = ProgressBar::new(n as u64/ BATCH_SIZE as u64);
    pb.set_style(ProgressStyle::default_bar()
    .template("fine-tune [{bar:40}] {pos}/{len} loss={msg}")
        .unwrap()
        .progress_chars("█░░"));

    let mut total_loss = 0f32;
    let mut batches = 0;

    for i in(0..n).step_by(BATCH_SIZE){
        let end = (i + BATCH_SIZE).min(n);
        let batch_images = images.narrow(0, i, end - i)?;
        let batch_labels = labels.narrow(0, i, end - i)?.to_dtype(DType::U32)?;

        let logits = model.forward(&batch_images)?;
        let batch_loss = loss::cross_entropy(&logits, &batch_labels)?;

        opt.backward_step(&batch_loss)?;

        total_loss += batch_loss.to_scalar::<f32>()?;
        batches += 1;
        pb.inc(1);
        pb.set_message(format!("{:.4}", total_loss / batches as f32));
    }

    pb.finish_with_message(format!("avg loss {:.4}",total_loss/batches as f32));
    Ok(())
}