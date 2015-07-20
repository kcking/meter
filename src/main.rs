#![feature(convert)]
#![feature(core)]
#![feature(split_off)]
#![feature(iter_arith)]

extern crate getopts;
extern crate core;

use getopts::Options;
use std::env;
use std::thread::{JoinHandle, Thread};
use std::thread;
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, BinaryHeap};
use std::sync::mpsc::{channel, Receiver, Sender};

extern crate bounded_spsc_queue;
use bounded_spsc_queue::{Producer, Consumer};

extern crate kissfft;
use kissfft::KissFFT;
use kissfft::binding::kiss_fft_cpx;

extern crate sound_stream;
//use sound_stream::{CallbackFlags, CallbackResult, SoundStream, Settings, StreamParams};
use sound_stream::*;
extern crate osc;
use osc::osc_sender::*;
use osc::osc_data::*;
use osc::osc_data::OscPacket::*;
use osc::osc_data::OscArg::*;
use std::net::{Ipv4Addr,SocketAddrV4};
use core::str::FromStr;

type ChannelBuffers = Arc<Mutex<HashMap<i32, Vec<f32>>>>;

const osc_interval : f32 = 1f32 / 30f32;

#[derive(Clone)]
struct IndexedMagnitude {
    index : usize,
    magnitude : f32
}

use std::cmp::Ordering;
impl PartialOrd for IndexedMagnitude {
     fn partial_cmp(&self, other: &IndexedMagnitude) -> Option<Ordering> {
        return self.magnitude.partial_cmp(&other.magnitude);
     }
}
impl PartialEq for IndexedMagnitude {
    fn eq(&self, other: &IndexedMagnitude) -> bool {
        return self.magnitude.eq(&other.magnitude);
    }
}
const Fs : usize = 44100usize;
extern crate num;
use num::integer::Integer;
fn pitch_detect(buckets : &Vec<f32>) -> f32{
    //  gcd of top 5 bands
    let mut indexed_buckets = vec!();
    let mut i = 0usize;
    for bucket in buckets {
        indexed_buckets.push(IndexedMagnitude{index : i, magnitude: bucket.clone()});
        i += 1usize;
    }
    let mut harmonics = indexed_buckets.clone();
    for indexed_bucket in indexed_buckets.iter() {
        for i in 1..4 {
            if indexed_bucket.index / i > 0usize {
                harmonics[indexed_bucket.index / i].magnitude += indexed_bucket.magnitude;
            }
        }
    }
    harmonics.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let max_idx = harmonics.pop().unwrap();
    //  convert index to freq
    let freq = (max_idx.index * Fs) as f32 / buckets.len() as f32;
    println!("{}", freq);
    return freq;
}

fn pitch_centroid(buckets : &Vec<f32>) -> f32 {
    let mut centroid = 0f32;
    let totalMag : f32 = buckets.iter().sum();
    let mut i = 1usize;
    for bucket in buckets {
        centroid += i as f32 * bucket / totalMag;
        i += 1usize;
    }
    centroid / buckets.len() as f32
}

use std::iter::FromIterator;
fn meter_fft(
    num_channels : i32,
    c : Consumer<f32>,
    osc_sender : Arc<Mutex<OscSender>>,
    osc_prefix : String
    ) -> JoinHandle<()> {
        let mut bufs_by_channel = HashMap::new();
        let mut chan_index = 0;
        let fft_buckets = 2048*4;
        thread::spawn(move || {
            loop {
                if let Some(s) = c.try_pop() {
                    let mut buf = bufs_by_channel.entry(chan_index).or_insert(vec!());
                    buf.push(kiss_fft_cpx{r : s, i : 0f32});
                    if buf.len() == fft_buckets {
                        let mut fft = KissFFT::new(fft_buckets, false);
                        let mut fft_out = fft.transform_to_vec(buf);
                        //  only use first half
                        fft_out.split_off(fft_buckets / 2usize);
                        let fft_norm = Vec::from_iter(fft_out.iter().map(|c| (c.r * c.r + c.i * c.i).sqrt()));
                        println!("pitch center {}", pitch_detect(&fft_norm));
                        println!("pitch centroid {}", pitch_centroid(&fft_norm));
                        buf.clear();
                    }

                    chan_index = (chan_index + 1) % num_channels;
                } else {
                    thread::sleep_ms(12u32 / num_channels as u32);
                }
            }
        })

    }
fn meter_rms(
    num_channels : i32,
    c : Consumer<f32>,
    p : Producer<f32>,
    osc_sender : Arc<Mutex<OscSender>>,
    osc_prefix : String
    ) -> JoinHandle<()> {
    let mut rms_map : HashMap<i32, f32> = HashMap::new();
    let mut chan_idx = 0;
    let mut time = 0f32;
    let mut last_sent_time = 0f32;
    // TODO: use sample rate
    let alpha = 30f32 / 44100f32;
    thread::spawn(move || {
        loop {
            if let Some(s) = c.try_pop() {
                let old_rms = *rms_map.entry(chan_idx).or_insert(0f32);
                let new_rms = ((old_rms.powi(2) + s.powi(2)) / 2f32).sqrt();
                rms_map.insert(chan_idx, (1f32 - alpha) * old_rms + new_rms * alpha);
                if time > last_sent_time + osc_interval {
                    let mut sender = osc_sender.lock().unwrap();
                    for (chan, rms) in rms_map.iter() {
                        let rms_msg = OscMessage{
                            addr : format!("/opera/meter/id-{}/track{}/rms", osc_prefix, chan).to_string(),
                            args : vec!(OscFloat(*rms as f32))
                        };
                        sender.send(rms_msg);
                    }
                    last_sent_time = time;
                }
                chan_idx = (chan_idx + 1) % num_channels;
                time += 1f32 / (44100f32 * num_channels as f32);
                p.push(s);
            } else {
                thread::sleep_ms(12u32 / num_channels as u32);
            }
        }
    })
}

fn setup_stream(
    frames_per_buffer : u16,
    num_channels : i32,
    p : Producer<f32>) {
     let f = Box::new(move |input: &[f32], _: Settings, o: &mut[f32], _: Settings, dt: f64, _: CallbackFlags| {
         for s in input.iter() {
             p.push(*s);
         }
         return CallbackResult::Continue;
     });

     let stream = SoundStream::new().frames_per_buffer(frames_per_buffer).duplex(StreamParams::new(), StreamParams::new()).run_callback(f).unwrap();
     while let Ok(true) = stream.is_active() {
        thread::sleep_ms(500);
     }
}

fn main() {
     let args: Vec<String> = env::args().collect();
     let num_channels : i32 = i32::from_str_radix(args[1].as_str(), 10).unwrap();
     let frames_per_buffer : usize = usize::from_str_radix(args[2].as_str(), 10).unwrap();
     let send_ip = args[3].as_str();
     let send_port = u16::from_str_radix(args[4].as_str(), 10).unwrap();
     let meter_id = args[5].as_str();

     let (p_pa, c_rms) = bounded_spsc_queue::make(16 * frames_per_buffer * (num_channels as usize));
     let (p_rms, c_fft) = bounded_spsc_queue::make(16 * frames_per_buffer * (num_channels as usize));

     let mut sender;
     match OscSender::new(
         SocketAddrV4::new(Ipv4Addr::from_str("127.0.0.1").unwrap(), 7000),
         SocketAddrV4::new(Ipv4Addr::from_str(send_ip).unwrap(), send_port)
         ) {
         Ok(s) => { sender = s; },
         Err(e) => { panic!(e); }
     }
     let sender_arc = Arc::new(Mutex::new(sender));
     meter_rms(num_channels, c_rms, p_rms, sender_arc.clone(), String::from(meter_id));
     meter_fft(num_channels, c_fft, sender_arc.clone(), String::from(meter_id));
     setup_stream(frames_per_buffer as u16, num_channels, p_pa);
}