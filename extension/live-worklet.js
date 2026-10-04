// Mixes the captured tab audio down to mono and decimates it to 16 kHz in 100 ms blocks.
class Downsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.phase = 0;
    this.acc = 0;
    this.n = 0;
    this.out = new Float32Array(1600);
    this.o = 0;
  }

  process(inputs) {
    const chans = inputs[0];
    if (!chans || !chans.length) return true;
    const len = chans[0].length;
    for (let i = 0; i < len; i++) {
      let s = 0;
      for (let c = 0; c < chans.length; c++) s += chans[c][i];
      this.acc += s / chans.length;
      this.n++;
      this.phase++;
      if (this.phase >= this.ratio) {
        this.phase -= this.ratio;
        this.out[this.o++] = this.acc / this.n;
        this.acc = 0;
        this.n = 0;
        if (this.o === this.out.length) {
          this.port.postMessage(this.out.buffer, [this.out.buffer]);
          this.out = new Float32Array(1600);
          this.o = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("downsampler", Downsampler);
