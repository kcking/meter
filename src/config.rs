extern crate toml;

use std::io::prelude::*;
use std::fs::File;
use std::collections::HashMap;

#[derive(RustcDecodable, Clone)]
pub struct MeterConfig {
    pub active_channels : HashMap<String, String>,
    pub total_channels : i32,
    pub frames_per_buffer : usize,
    pub send_ip : String,
    pub send_port : u16,
    pub meter_id : String,
    pub show_graphics : bool,
    pub sampling_frequency : u32,
}

pub fn get_config(toml_file : &str) -> MeterConfig {
    let mut f = File::open(toml_file).unwrap();
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    toml::decode_str(s.as_str()).unwrap()
}
