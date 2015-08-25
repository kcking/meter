#Install portaudio and make sure pkg-config can find it
	pkg-config --libs portaudio-2.0
#Install Rust Nightly 2015-08-07
    go to https://static.rust-lang.org/dist/2015-08-07/index.html and find the
    nightly installer for your OS.
##Mac
    https://static.rust-lang.org/dist/2015-08-07/rust-nightly-x86_64-apple-darwin.pkg
#Run Meter (in meter directory)
    cargo run <num_active_channels> <num_total_channels> <frames/buffer> <osc sendip> <osc sendport> <unique
    id for server> <Optional<bool> show graphics> --release
#Features
    - Moving average RMS
##Alpha
    - pitch centroid
    - monophonic pitch detection
##Upcoming
    - Consonance
