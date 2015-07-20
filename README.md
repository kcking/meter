#Install Rust
    curl -sSf https://static.rust-lang.org/rustup.sh | sh -s -- --channel=nightly
#Run Meter (in meter directory)
    cargo run <num_channels> <frames/buffer> <osc sendip> <osc sendport> <unique
    id for server> --release