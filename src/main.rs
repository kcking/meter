#![feature(convert)]
#![feature(core)]
#![feature(split_off)]
#![feature(vec_push_all)]

extern crate core;
extern crate rustc_serialize;

extern crate piston;
extern crate graphics;
extern crate glutin_window;
extern crate opengl_graphics;

mod display;
mod config;
mod frequency_domain;
use frequency_domain::*;

use std::env;
use std::thread::{JoinHandle};
use std::thread;
use std::sync::{Arc, Mutex};
use std::collections::{HashMap};

extern crate bounded_spsc_queue;
use bounded_spsc_queue::{Producer, Consumer};

extern crate kissfft;
extern crate sound_stream;
//use sound_stream::{CallbackFlags, CallbackResult, SoundStream, Settings, StreamParams};
use sound_stream::*;
extern crate osc;
use osc::osc_sender::*;
use osc::osc_data::OscPacket::*;
use osc::osc_data::OscArg::*;
use std::net::{Ipv4Addr,SocketAddrV4};
use core::str::FromStr;

type ChannelBuffers = Arc<Mutex<HashMap<i32, Vec<f32>>>>;

const OSC_PER_SEC : u32 = 30u32;

#[macro_use]
extern crate itertools;



fn send_rms_osc(
    sender : &Arc<Mutex<OscSender>>,
    active_channels : &[String],
    rms_list : &[f32],
    max_amp_list : &[f32],
    osc_prefix : &String,
    ) {

        let mut sender = sender.lock().unwrap();
        for i in 0..active_channels.len() {
            let track_title = active_channels[i].clone();
            let rms = rms_list[i];
            let max_amp = max_amp_list[i];
            let rms_msg = OscMessage{
                addr : format!("/{}/{}/rms", track_title, osc_prefix).to_string(),
                args : vec!(OscFloat(rms as f32))
            };
            if let Err(e) = sender.send(rms_msg) {
                println!("Error sending OSC: {:?}", e);
            }
            let amp_msg = OscMessage{
                addr : format!("/{}/{}/maxAmp", track_title, osc_prefix).to_string(),
                args : vec!(OscFloat(max_amp))
            };
            if let Err(e) = sender.send(amp_msg) {
                println!("Error sending OSC: {:?}", e);
            }
        }
    }

fn meter_rms(
    sampling_frequency : u32,
    active_channels : Vec<String>,
    c : Consumer<f32>,
    p : Producer<f32>,
    osc_sender : Arc<Mutex<OscSender>>,
    osc_prefix : String,
    ) -> JoinHandle<()> {
        let mut rms_vec : Vec<f32> = Vec::new();
        let mut max_amp_vec : Vec<f32> = Vec::new();
        for _ in 0..active_channels.len() {
            rms_vec.push(0.);
            max_amp_vec.push(0.);
        }
        let mut chan_idx = 0;
        let mut samples = 0usize;
        let mut last_sent_time = 0usize;
        let alpha = 30f32 / sampling_frequency as f32;
        thread::spawn(move || {
            loop {
                if let Some(s) = c.try_pop() {
                    let old_rms = rms_vec[chan_idx].clone();
                    let new_rms = ((old_rms.powi(2) + s.powi(2)) / 2f32).sqrt();
                    rms_vec[chan_idx] = (1f32 - alpha) * old_rms + new_rms * alpha;

                    let old_max_amp = max_amp_vec[chan_idx].clone();
                    max_amp_vec[chan_idx] = f32::max(old_max_amp, f32::abs(s));
                    if samples > last_sent_time + (sampling_frequency / active_channels.len() as u32 / OSC_PER_SEC) as usize {
                        send_rms_osc(&osc_sender, &active_channels, &rms_vec, &max_amp_vec, &osc_prefix);
                        for i in 0..active_channels.len() {
                            max_amp_vec[i] = 0.;
                        }
                        last_sent_time = samples;
                    }
                    samples += 1usize;
                    chan_idx = (chan_idx + 1) % active_channels.len();
                    p.push(s);
                } else {
                    thread::sleep_ms(1u32);
                }
            }
        })
    }

fn setup_stream(
    sampling_frequency : u32,
    frames_per_buffer : u16,
    channel_active_vec : Vec<bool>,
    total_channels : i32,
    p : Producer<f32>) {
        let f = Box::new(move |input: &[f32], _: Settings, _ : &mut[f32], _: Settings, _: f64, _: CallbackFlags| {
            let mut channel_index = 0usize;
            for s in input.iter() {
                if channel_active_vec[channel_index] {
                    p.push(*s);
                }
                channel_index = (channel_index + 1usize) % total_channels as usize;
            }
            return CallbackResult::Continue;
        });

        let stream = SoundStream::new().sample_hz(sampling_frequency as f64).frames_per_buffer(frames_per_buffer).duplex(StreamParams::new().channels(total_channels), StreamParams::new()).run_callback(f).unwrap();
        while let Ok(true) = stream.is_active() {
            thread::sleep_ms(500);
        }
    }
use config::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config_file = "config/default.toml";
    if args.len() > 1 {
        config_file = args[1].as_str();
    }
    let config = config::get_config(config_file);
    let active_channels_strings = config.active_channels.clone();
    let mut active_channels : HashMap<i32, String> = HashMap::new();
    //  convert map from <String, String> to <i32, String> because toml doesn't support <i32, String>
    for (chan, name) in active_channels_strings.into_iter() {
        active_channels.insert(i32::from_str_radix(&chan, 10).unwrap(), name);
    }

    let total_channels = config.total_channels;
    let mut channel_active_vec = Vec::new();
    let mut active_channel_collapsed_titles_vec = vec!();
    for i in 0..total_channels {
        channel_active_vec.push(active_channels.contains_key(&i));
        if let Some(title) = active_channels.get(&i) {
            active_channel_collapsed_titles_vec.push(title.clone());
        }
    }
    let frames_per_buffer = config.frames_per_buffer;
    let send_ip = config.send_ip.clone();
    let send_port = config.send_port;
    let meter_id = config.meter_id.clone();
    let show_graphics = config.show_graphics;

    let (p_pa, c_rms) = bounded_spsc_queue::make(16 * frames_per_buffer * (total_channels as usize));
    let (p_rms, c_fft) = bounded_spsc_queue::make(16 * frames_per_buffer * (total_channels as usize));

    let sender =
        match OscSender::new(
            SocketAddrV4::new(Ipv4Addr::from_str("0.0.0.0").unwrap(), 0),
            SocketAddrV4::new(Ipv4Addr::from_str(send_ip.as_str()).unwrap(), send_port)
            ) {
                Ok(s) => { s },
                Err(e) => { panic!(e); }
            };
    let sender_arc = Arc::new(Mutex::new(sender));
    let fft_magnitudes = Arc::new(Mutex::new(vec!()));
    let vertical_lines = Arc::new(Mutex::new(vec!()));
    meter_rms(config.sampling_frequency.clone(), active_channel_collapsed_titles_vec.clone(), c_rms, p_rms, sender_arc.clone(), meter_id.clone());
    meter_fft(config.sampling_frequency as usize, active_channel_collapsed_titles_vec.clone(), c_fft, sender_arc.clone(), meter_id, fft_magnitudes.clone(), vertical_lines.clone());
    let config_setup = config.clone();
    thread::spawn(move || setup_stream(config_setup.sampling_frequency, frames_per_buffer as u16, channel_active_vec, total_channels, p_pa));
    if show_graphics {
        display::init(fft_magnitudes, vertical_lines);
    }
    loop {
        thread::sleep_ms(500);
    }
}
