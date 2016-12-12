#Meter
	Monitor frequency and amplitude of audio input, send realtime measurements
	over OSC.

##Install portaudio and make sure pkg-config can find it
	pkg-config --libs portaudio-2.0
##Install Rust Nightly With rustup
	https://www.rustup.rs/
##Run Meter (in meter directory)
    cargo run [config_file] --release
