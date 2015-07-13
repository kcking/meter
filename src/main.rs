#![feature(convert)]

extern crate getopts;

use getopts::Options;
use std::env;
use std::thread::{JoinHandle, Thread};
use std::thread;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};


extern crate sound_stream;
use sound_stream::{CallbackFlags, CallbackResult, SoundStream, Settings, StreamParams};

extern crate osc;
use osc::osc_sender::*;
use osc::osc_data::*;
use std::io::net::ip::{Ipv4Addr,SocketAddr};

type ChannelBuffers = Arc<Mutex<HashMap<i32, Vec<f32>>>>;

fn meter_rms(bufs : ChannelBuffers, rx : Receiver<()>) -> JoinHandle<()> {
    thread::spawn(move || {
        while let Ok(_) = rx.recv() {
            let mut rms_map : HashMap<i32, f32> = HashMap::new();
            for (chan, samples) in bufs.lock().unwrap().iter_mut() {
                let mut rms = 0.0;
                for s in samples.iter() {
                    rms += *s;
                }
                rms /= samples.len() as f32;
                rms_map.insert(chan, rms);
                samples.clear();
            }

        }
    })
}

fn setup_stream(frames_per_buffer : u16, num_channels : i32, bufs : ChannelBuffers, tx : Sender<()>) {
     let mut channel_idx = 0;
     let f = Box::new(move |input: &[f32], _: Settings, o: &mut[f32], _: Settings, dt: f64, _: CallbackFlags| {
         let mut bufs = bufs.lock().unwrap();
         for s in input.iter() {
             bufs.entry(channel_idx).or_insert(vec!()).push(*s);
             channel_idx = (channel_idx + 1) % num_channels;
         }
         tx.send(());
         CallbackResult::Continue
     });

     let stream = SoundStream::new().frames_per_buffer(frames_per_buffer).duplex(StreamParams::new().channels(num_channels), StreamParams::new()).run_callback(f).unwrap();

}

fn main() {
     let args: Vec<String> = env::args().collect();
     let num_channels : i32 = i32::from_str_radix(args[1].as_str(), 10).unwrap();
     let frames_per_buffer : usize = usize::from_str_radix(args[2].as_str(), 10).unwrap();

     let buffersByChannels : Arc<Mutex<HashMap<i32, Vec<f32>>>> = Arc::new(Mutex::new(HashMap::new()));

     let (tx, rx) = channel();
     setup_stream(frames_per_buffer as u16, num_channels, buffersByChannels.clone(), tx);
     meter_rms(buffersByChannels.clone(), rx);

     loop {
        thread::sleep_ms(500);
     }

}
