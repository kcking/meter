#Install Rust
    curl -sSf https://static.rust-lang.org/rustup.sh | sh -s -- --channel=nightly
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