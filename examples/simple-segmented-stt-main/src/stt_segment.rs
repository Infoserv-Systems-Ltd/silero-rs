use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct STTSegment {
    start: i64,
    end: i64,
    text: String,
}

impl STTSegment {
    pub fn new(start: i64, end: i64, text: String) -> Self {
        STTSegment { start, end, text }
    }

    pub fn start(&self) -> i64 {
        self.start
    }

    pub fn start_as_timestamp(&self) -> String {
        STTSegment::to_timestamp(self.start)
    }

    pub fn end(&self) -> i64 {
        self.end
    }

    pub fn end_as_timestamp(&self) -> String {
        STTSegment::to_timestamp(self.end)
    }

    pub fn text(&self) -> &str {
        self.text.as_str()
    }

    fn to_timestamp(timestamp: i64) -> String {
        let sec = timestamp / 100;
        let msec = timestamp - sec * 100;
        let min = sec / 60;
        let sec = sec - min * 60;

        format!("{:02}:{:02}:{:03}", min, sec, msec)
    }
}
