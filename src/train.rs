use candle_core::{DType, Result };
use candle_nn::{loss, Optimizer, VarMap};
use indicatif::{ProgressBar, ProgressStyle};
use crate::model::MLP;
use crate::data::MnistData;

const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 5;
const BATCH_SIZE: usize = 64;

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