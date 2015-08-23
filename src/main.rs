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

#[allow(non_upper_case_globals)]
const Fs : usize = 44100usize;
const OSC_INTERVAL : usize = Fs / 30usize;

#[macro_use]
extern crate itertools;


fn meter_rms(
    num_channels : i32,
    c : Consumer<f32>,
    p : Producer<f32>,
    osc_sender : Arc<Mutex<OscSender>>,
    osc_prefix : String
    ) -> JoinHandle<()> {
    let mut rms_map : HashMap<i32, f32> = HashMap::new();
    let mut max_amp_map : HashMap<i32, f32> = HashMap::new();
    let mut chan_idx = 0;
    let mut samples = 0usize;
    let mut last_sent_time = 0usize;
    // TODO: use sample rate
    let alpha = 30f32 / 44100f32;
    thread::spawn(move || {
        loop {
            if let Some(s) = c.try_pop() {
                let old_rms = *rms_map.entry(chan_idx).or_insert(0f32);
                let new_rms = ((old_rms.powi(2) + s.powi(2)) / 2f32).sqrt();
                rms_map.insert(chan_idx, (1f32 - alpha) * old_rms + new_rms * alpha);

                let old_max_amp = *max_amp_map.entry(chan_idx).or_insert(0f32);
                max_amp_map.insert(chan_idx, f32::max(old_max_amp, f32::abs(s)));
                if samples > last_sent_time + OSC_INTERVAL * (num_channels as usize) {
                    let mut sender = osc_sender.lock().unwrap();
                    let mut msg_vec = vec!();
                    for (chan, rms) in rms_map.iter() {
                        let rms_msg = OscMessage{
                            addr : format!("/opera/meter/{}/track{}/rms", osc_prefix, chan).to_string(),
                            args : vec!(OscFloat(*rms as f32))
                        };
                        msg_vec.push(rms_msg);
                    }
                    for (chan, max_amp) in max_amp_map.iter() {
                        let amp_msg = OscMessage{
                            addr : format!("/opera/meter/{}/track{}/maxAmp", osc_prefix, chan).to_string(),
                            args : vec!(OscFloat(*max_amp))
                        };
                        msg_vec.push(amp_msg);
                    }
                    max_amp_map.clear();
                    let bundle = OscBundle {
                        time_tag : (0, 1),
                        conts : msg_vec
                    };
                    if let Err(e) = sender.send(bundle) {
                        println!("Error sending OSC: {:?}", e);
                    }
                    last_sent_time = samples;
                }
                chan_idx = (chan_idx + 1) % num_channels;
                samples += 1usize;
                p.push(s);
            } else {
                thread::sleep_ms(1u32);
            }
        }
    })
}

fn setup_stream(
    frames_per_buffer : u16,
    active_channels : i32,
    total_channels : i32,
    p : Producer<f32>) {
        let mut channel_idx = 0usize;
        let f = Box::new(move |input: &[f32], _: Settings, _ : &mut[f32], _: Settings, _: f64, _: CallbackFlags| {
            for s in input.iter() {
                if (channel_idx as i32) < active_channels {
                    p.push(*s);
                }
                channel_idx = (channel_idx + 1usize) % total_channels as usize;
            }
            return CallbackResult::Continue;
        });

        let stream = SoundStream::new().sample_hz(Fs as f64).frames_per_buffer(frames_per_buffer).duplex(StreamParams::new().channels(total_channels), StreamParams::new()).run_callback(f).unwrap();
        while let Ok(true) = stream.is_active() {
            thread::sleep_ms(500);
        }
}
use config::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config_file = "meter.toml";
    if args.len() > 1 {
        config_file = args[1].as_str();
    }
    let config = config::get_config(config_file);
    let active_channels = config.active_channels;
    let total_channels = config.total_channels;
    let frames_per_buffer = config.frames_per_buffer;
    let send_ip = config.send_ip;
    let send_port = config.send_port;
    let meter_id = config.meter_id;
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
    meter_rms(active_channels, c_rms, p_rms, sender_arc.clone(), meter_id.clone());
    meter_fft(active_channels, c_fft, sender_arc.clone(), meter_id, fft_magnitudes.clone(), vertical_lines.clone());
    thread::spawn(move || setup_stream(frames_per_buffer as u16, active_channels, total_channels, p_pa));
    if show_graphics {
        display::init(fft_magnitudes, vertical_lines);
    }
    loop {
        thread::sleep_ms(500);
    }
}
