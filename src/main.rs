mod data;

use candle_core::Device;


fn main()->candle_core::Result<()> {

    let device = Device::Cpu;
    let mnist = data::load_mnist(&device)?;

    println!("Train img shape: {:?}", mnist.train_images.shape());
    println!("Test img shape: {:?}", mnist.test_images.shape());

    Ok(())
}
