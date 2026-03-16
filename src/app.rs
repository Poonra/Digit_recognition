use eframe::egui;
use egui:: {Color32,Pos2,Stroke,Vec2};
use candle_core::{Device,Tensor};
use crate::model::MLP;
use std::fs;

#[derive(serde::Serialize, serde::Deserialize)]
struct Sample {
    pixels:Vec<f32>,
    label:u32,
}

pub struct DrawingApp {
    canvas_size:f32,
    stroke: Vec<Vec<Pos2>>,
    current_stroke: Vec<Pos2>,
    prediction: Option<u32>,
    probs: Vec<f32>,
    model:MLP,
    device: Device,

    last_pixels: Option<Vec<f32>>,  
    show_correction: bool,        
    collected: Vec<Sample>,         // all samples collected 
    saved_count: usize,             // how many we saved total
}

impl DrawingApp {
    pub fn new(model:MLP, device: Device) -> Self{

        //load existing sample
        let collected = if let Ok(data)=  fs::read_to_string("collected_data.json"){
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        let saved_count = collected.len();

        Self{
            canvas_size:280.0, //10x scale of 28px
            stroke:Vec::new(),
            current_stroke: Vec::new(),
            prediction: None,
            probs:vec![0.0; 10],
            model,
            device,
            last_pixels: None,
            show_correction:false,
            collected,
            saved_count,

    
        }

    }
    fn predict(&mut self){
        //create blank canvas
        let mut pixels = vec![0f32; 28*28];
        let scale = self.canvas_size /28.0;

        //drow stroke on pixel grid
        for stroke in &self.stroke{
            for window in stroke.windows(2){
                let p0 = window[0];
                let p1 = window[1];

                //convert screen coord to pixel coords

                let x0 = (p0.x/scale) as i32;
                let y0 = (p0.y/scale) as i32;
                let x1 = (p1.x/scale) as i32;
                let y1 = (p1.y/scale) as i32;

                // draw thick dot at point pixel
                for dx in -2..=2i32{
                    for dy in -2..= 2i32{
                    
                    let px = (x0+dx).clamp(0,27) as usize;
                    let py = (y0+dy).clamp(0,27) as usize;
                    pixels[py*28+px] = 1.0;
                    let px = (x1+dx).clamp(0,27) as usize;
                    let py = (y1+dy).clamp(0,27) as usize;
                    pixels[py*28+px] = 1.0;
                    }
                }
            }
        }

        //save pixel
        self.last_pixels= Some(pixels.clone());
        self.show_correction = false;
    

    //build tensor
    let tensor = Tensor::from_vec(pixels, (1,784),&self.device).unwrap();
    let logits = self.model.forward(&tensor).unwrap();
    let probs = candle_nn::ops::softmax(&logits,1).unwrap();
    let probs = probs.squeeze(0).unwrap();

    self.prediction= Some(probs.argmax(0).unwrap().to_scalar::<u32>().unwrap());
    self.probs = (0..10)
        .map(|i| probs.get(i).unwrap().to_scalar::<f32>().unwrap())
        .collect();

    }

    fn save_sample(&mut self,label:u32){
        if let Some(pixels) = self.last_pixels.take(){
            self.collected.push(Sample{pixels,label});
            
            let json = serde_json::to_string(&self.collected).unwrap();
            fs::write("collected_data.json", json).unwrap();

            self.saved_count= self.collected.len();
            self.show_correction = false;
            println!("Saved sample with label {}",label);

        }
    }
}

impl eframe::App for DrawingApp{
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx,|ui|{
            ui.heading("Draw a digit");
            ui.horizontal(|ui|{

                //draw canvas
                let (response, painter) = ui.allocate_painter(
                    Vec2::splat(self.canvas_size),
                    egui::Sense::drag(),
                );
                 // black background
                painter.rect_filled(response.rect, 0.0, Color32::BLACK);

                    // collect mouse drag
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let local = pos - response.rect.min;
                        self.current_stroke.push(Pos2::new(local.x, local.y));
                    }
                }
                    // when drag end save stroke
                if response.drag_stopped() {
                    let stroke = std::mem::take(&mut self.current_stroke);
                    if !stroke.is_empty() {
                        self.stroke.push(stroke);
                    }
                }

                // draw all saved stroke
                for stroke in &self.stroke {
                    for window in stroke.windows(2) {
                        painter.line_segment(
                            [response.rect.min + window[0].to_vec2(),
                             response.rect.min + window[1].to_vec2()],
                            Stroke::new(18.0, Color32::WHITE),
                        );
                    }
                }
                // draw current stroke
                for window in self.current_stroke.windows(2) {
                    painter.line_segment(
                        [response.rect.min + window[0].to_vec2(),
                         response.rect.min + window[1].to_vec2()],
                        Stroke::new(18.0, Color32::WHITE),
                    );
                }

                 // Sidebar: buttons + results 
                ui.vertical(|ui| {
                    if ui.button("Predict").clicked() {
                        self.predict();
                    }
                    if ui.button("Clear").clicked() {
                        self.stroke.clear();
                        self.current_stroke.clear();
                        self.prediction = None;
                        self.probs = vec![0.0; 10];
                    }
                    //prediction
                    if let Some(pred) = self.prediction {
                        ui.label(format!("Predicted: {}", pred));
                        ui.separator();
                        for (digit, &p) in self.probs.iter().enumerate() {
                            ui.label(format!("{}: {:5.1}%", digit, p * 100.0));
                        }
                        ui.separator();
                       // correct/wrong btn
                    if !self.show_correction {
                            ui.label("Was this correct?");
                            ui.horizontal(|ui| {
                                if ui.button("✓ Yes").clicked() {
                                    self.save_sample(pred);
                                    self.stroke.clear();
                                    self.prediction = None;
                                    self.probs.clear();
                                }
                                if ui.button("✗ No").clicked() {
                                    self.show_correction = true;
                                }
                            });
                    }
                    if self.show_correction {
                        ui.label("Select the correct digit:");
                        ui.horizontal(|ui| {
                            for digit in 0..=4u32 {
                                if ui.button(digit.to_string()).clicked() {
                                    self.save_sample(digit);  // save with corrected label
                                    self.stroke.clear();
                                    self.prediction = None;
                                    self.probs.clear();
                                }
                            }
                         });
                            

                             ui.horizontal(|ui| {
                                for digit in 5..=9u32 {
                                    if ui.button(digit.to_string()).clicked() {
                                        self.save_sample(digit);
                                        self.stroke.clear();
                                        self.prediction = None;
                                        self.probs.clear();
                                    }
                                }
                            });
                        }
    

                    }
                });
            });
        });
    }
}