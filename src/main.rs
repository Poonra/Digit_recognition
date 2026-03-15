mod data;
mod model;
use candle_core::Device;
use candle_nn::{VarBuilder,VarMap};


fn main()->candle_core::Result<()> {

    let device = Device::Cpu;
    let mnist = data::load_mnist(&device)?;

    println!("Train img shape: {:?}", mnist.train_images.shape());
    println!("Test img shape: {:?}", mnist.test_images.shape());

    //varmap keep all trainable parameter

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap,candle_core::DType::F32, &device);
    let model = model::MLP::new(vb)?;

    // test a forward pass with first 4 img
    let sample =mnist.train_images.narrow(0,0,4)?;
    let output = model.forward(&sample)?;
    println!("Output shape: {:?}",output.shape());

    Ok(())
}
