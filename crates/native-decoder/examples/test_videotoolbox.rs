use native_decoder::{DecoderConfig, NativeVideoDecoder, VideoToolboxDecoder};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("Testing VideoToolbox implementation...");

    // Test with a sample video file path (user should replace with actual video)
    let video_path = "/Users/mingeonkim/LocalDocuments/Gausian/gausian.ai.frontend/react-app/public/media/shot_01.mp4".to_string();

    if !Path::new(&video_path).exists() {
        println!("Please update the video_path in this example to point to a real video file");
        println!("Current path: {}", video_path);
        return Ok(());
    }

    // Create decoder config
    let config = DecoderConfig {
        hardware_acceleration: true,
        preferred_format: None,
        zero_copy: true, // Test zero-copy IOSurface decoding
    };

    // Create VideoToolbox decoder
    println!("Creating VideoToolbox decoder...");
    let mut decoder = VideoToolboxDecoder::new(video_path, config)?;

    println!("Decoder created successfully!");
    println!("Video properties:");
    let props = decoder.get_properties();
    println!("  Width: {}", props.width);
    println!("  Height: {}", props.height);
    println!("  Duration: {:.2}s", props.duration);
    println!("  Frame rate: {:.2} fps", props.frame_rate);
    println!("  Format: {:?}", props.format);

    // Test CPU decoding
    println!("Testing CPU decoding...");

    match decoder.decode_frame(1.0) {
        Ok(Some(frame)) => println!(
            "Successfully decoded CPU frame at 1.0s: {}x{}",
            frame.width, frame.height
        ),
        Ok(None) => println!("No frame available at timestamp 1.0s"),
        Err(e) => println!("CPU decoding failed: {}", e),
    }

    println!("\nTesting zero-copy IOSurface decoding...");
    match decoder.decode_frame_zero_copy(2.0) {
        Ok(Some(frame)) => println!(
            "Successfully decoded IOSurface frame at 2.0s: {}x{}",
            frame.width, frame.height
        ),
        Ok(None) => println!("No IOSurface frame available at timestamp 2.0s"),
        Err(e) => println!("IOSurface decoding failed: {}", e),
    }

    // Test multiple frame decoding
    println!("\nTesting sequential frame decoding...");
    for i in 0..5 {
        let timestamp = i as f64 * 0.5; // Every 0.5 seconds
        if let Ok(Some(frame)) = decoder.decode_frame(timestamp) {
            println!(
                "CPU frame at {}s: {}x{}",
                timestamp, frame.width, frame.height
            );
        }
        if let Ok(Some(frame)) = decoder.decode_frame_zero_copy(timestamp) {
            println!(
                "IOSurface frame at {}s: {}x{}",
                timestamp, frame.width, frame.height
            );
        }
    }

    println!("\nVideoToolbox test completed!");
    Ok(())
}
