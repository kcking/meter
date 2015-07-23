#Install Rust Nightly 2015-07-21
    go to https://static.rust-lang.org/dist/2015-07-21/index.html and find the
    installer for your OS.
##Mac
    https://static.rust-lang.org/dist/2015-07-21/rust-nightly-i686-apple-darwin.pkg
#Run Meter (in meter directory)
    cargo run <num_active_channels> <num_total_channels> <frames/buffer> <osc sendip> <osc sendport> <unique
    id for server> --release
#Features
    - Moving average RMS
##Alpha
    - pitch centroid
    - monophonic pitch detection
##Upcoming
    - Consonance