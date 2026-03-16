mod data;
mod model;
mod train;
mod eval;
mod predict;
mod app;

use candle_core::Device;
use candle_nn::{VarBuilder,VarMap};


fn main()->candle_core::Result<()> {

    let device = Device::Cpu;
    let mnist = data::load_mnist(&device)?;

    println!("Data loaded");
   

    //varmap keep all trainable parameter

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap,candle_core::DType::F32, &device);
    let model = model::MLP::new(vb)?;
    println!("Model ready!");

    let weights_path ="model_weights.safetensors";

    if std::path::Path::new(weights_path).exists(){
        // weights file found 
        println!("Found saved weights, loading...");
        varmap.load(weights_path)?;
        println!("Weights loaded!");
    } else{

    train::train(&mnist, &varmap, &model)?;
    println!("training complete");

    train::finetune_on_collected(&varmap, &model, &device)?;

    varmap.save(weights_path)?;
    println!("Weights saved to {}", weights_path);
    }
    eval::evaluate(&mnist, &model)?;

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Digit Recognition",
        native_options,
        Box::new(|_cc| Ok(Box::new(app::DrawingApp::new(model, device)))),
    ).unwrap();

    //predict::predict(&model, &test_image, true_label)?;
    Ok(())
}
