use candle_core::{Device, Result, Tensor};
use std::fs::File;
use std::io::Read;

pub struct MnistData{
    pub train_images: Tensor, // MNIST training 60k images and 784 pixel each
    pub train_labels: Tensor, //shape 60k
    pub test_images: Tensor,// 10k test img and 784 pixel each
    pub test_labels: Tensor, //Mnist has 10k test img
}

fn read_labels(path: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut header = [0u8;8];
    file.read_exact(&mut header)?;
    let mut labels = Vec::new();
    file.read_to_end(&mut labels)?;
    Ok(labels)
}

fn read_images(path:&str) -> Result<Vec<f32>>{
    let mut file = File::open(path)?;
    let mut header = [0u8;16];
    file.read_exact(&mut header)?; // skip magic number
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    // normalize pixel from 0-255 to 0.0 - 1.0
    Ok(bytes.iter().map(|&b| b as f32/255.0).collect())
}

pub fn load_mnist(device:&Device) -> Result<MnistData> {

  let train_imgs = read_images("data/train-images.idx3-ubyte").unwrap();
    let train_lbls = read_labels("data/train-labels.idx1-ubyte").unwrap();
    let test_imgs  = read_images("data/t10k-images.idx3-ubyte").unwrap();
    let test_lbls  = read_labels("data/t10k-labels.idx1-ubyte").unwrap();
    
    Ok(MnistData {
        train_images:Tensor::from_vec(train_imgs,(60000,784),device)?,
        train_labels:Tensor::from_vec(train_lbls,(60000,),device)?,
        test_images:Tensor::from_vec(test_imgs,(10000,784),device)?,
        test_labels:Tensor::from_vec(test_lbls,(10000,),device)?,
    })
}