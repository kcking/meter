#Install portaudio and make sure pkg-config can find it
	pkg-config --libs portaudio-2.0
#Install Rust Nightly 2015-08-23
    go to https://static.rust-lang.org/dist/2015-08-23/index.html and find the
    nightly installer for your OS.
##Mac
    https://static.rust-lang.org/dist/2015-08-23/rust-nightly-x86_64-apple-darwin.pkg
#Run Meter (in meter directory)
    cargo run [config_file] --release
#Features
    - Moving average RMS
##Alpha
    - pitch centroid
    - monophonic pitch detection
##Upcoming
    - Consonance
