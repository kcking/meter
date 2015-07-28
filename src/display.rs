
use piston::window::WindowSettings;
use piston::event::*;
use glutin_window::GlutinWindow as Window;
use opengl_graphics::{ GlGraphics, OpenGL };

use std::sync::{Arc, Mutex};

pub struct App {
    // OpenGL drawing backend.
    gl: GlGraphics,
    fft_magnitudes: Arc<Mutex<Vec<f32>>>,
    vertical_lines : Arc<Mutex<Vec<([f32 ; 4], usize)>>>
}

impl App {
    pub fn set_fft_magnitudes(&mut self, mags : &Vec<f32>) {
        let mut app_mags = self.fft_magnitudes.lock().unwrap();
        app_mags.clear();
        for f in mags.iter() {
            app_mags.push(*f);
        }
    }
}

impl App {
    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;

        const WHITE: [f32; 4] = [231.0/255.0, 222.0/255.0, 219.0/255.0, 1.0];
        const BLUE:   [f32; 4] = [18.0/255.0, 87.0/255.0, 150.0/255.0, 1.0];
        let fft_magnitudes = self.fft_magnitudes.lock().unwrap();
        let vertical_lines = self.vertical_lines.lock().unwrap();
        let unit_square = rectangle::square(0.0, 0.0, 1.0);
        let (x, y) = ((args.width / 2) as f64, (args.height / 2) as f64);
        if fft_magnitudes.len() == 0{
            return;
        }

        let N = f64::min(fft_magnitudes.len() as f64, 512.);

        self.gl.draw(args.viewport(), |c, gl| {
            let w = args.width;
            let h = args.height;
            let side_length = w as f64 / N;

            // Clear the screen.
            clear(BLUE, gl);

            // Draw a box rotating around the middle of the screen.
            let mut i = 0;
            for fft_mag in fft_magnitudes.iter() {
                if i as f64 >= N {
                    break;
                }
                rectangle(
                    WHITE,
                    [i as f64 * side_length, 0.0, side_length, *fft_mag as f64 * h as f64 /2.],
                    c.transform,
                    gl);
                i += 1;
            }
            for &(color, vert_idx) in vertical_lines.iter() {
                rectangle(
                    color,
                    [vert_idx as f64 / N as f64 * w as f64, 0.0, 1.0, h as f64],
                    c.transform,
                    gl);
            }
        });
    }

    fn update(&mut self, args: &UpdateArgs) {
    }
}

pub fn init(fft_magnitudes : Arc<Mutex<Vec<f32>>>,
            vertical_lines : Arc<Mutex<Vec<([f32; 4], usize)>>>) {
    let opengl = OpenGL::V3_2;

    // Create an Glutin window.
    let window = Window::new(
        WindowSettings::new(
            "meter | Opera of the Future",
            [920, 500]
        )
        .opengl(opengl)
        .exit_on_esc(true)
    );

    // Create a new game and run it.
    let mut app = App {
        gl: GlGraphics::new(opengl),
        fft_magnitudes: fft_magnitudes,
        vertical_lines : vertical_lines
    };
    for e in window.events() {
        if let Some(r) = e.render_args() {
            app.render(&r);
        }

        if let Some(u) = e.update_args() {
            app.update(&u);
        }
    }
}
