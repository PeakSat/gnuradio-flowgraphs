#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: SatNOGS-COMMS FSK BER test
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import blocks, gr
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, pdu
from gnuradio.filter import pfb
import gnuradio.satnogs as satnogs
import math
import numpy
import sip



class fsk_ber(gr.top_block, Qt.QWidget):

    def __init__(self, baudrate=50e3, delay_ms=100, frame_size=252, nframes=1000, samp_rate=2e6, sps_tx=16, tx_freq=435e6, tx_gain=60.0):
        gr.top_block.__init__(self, "SatNOGS-COMMS FSK BER test", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("SatNOGS-COMMS FSK BER test")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "fsk_ber")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Parameters
        ##################################################
        self.baudrate = baudrate
        self.delay_ms = delay_ms
        self.frame_size = frame_size
        self.nframes = nframes
        self.samp_rate = samp_rate
        self.sps_tx = sps_tx
        self.tx_freq = tx_freq
        self.tx_gain = tx_gain

        ##################################################
        # Variables
        ##################################################
        self.sq_wave = sq_wave = (1.0, ) * sps_tx
        self.gaussian_taps = gaussian_taps = filter.firdes.gaussian(1.0, sps_tx, 1.0, 4*sps_tx)
        self.deviation = deviation = 25e3
        self.variable_ieee802_15_4_variant_decoder_0 = variable_ieee802_15_4_variant_decoder_0 = satnogs.ieee802_15_4_variant_decoder([0b01010101]*12, 0, [0x1A, 0xCF, 0xFC, 0x1D], 4, satnogs.crc.CRC32_C, satnogs.whitening.make_ccsds(True), False, (448-4), True, False)
        self.variable_ieee802_15_4_encoder_0 = variable_ieee802_15_4_encoder_0 = satnogs.ieee802_15_4_encoder(0b01010101, 16, [0x1A, 0xCF, 0xFC, 0x1D], satnogs.crc.CRC32_C, satnogs.whitening.make_none(), False, False)
        self.modulation_index = modulation_index = deviation / (baudrate / 2.0)
        self.interp_taps = interp_taps = numpy.convolve(numpy.array(gaussian_taps), numpy.array(sq_wave))

        ##################################################
        # Blocks
        ##################################################

        self.satnogs_frame_encoder_0 = satnogs.frame_encoder(variable_ieee802_15_4_encoder_0)
        self.satnogs_frame_decoder_0_0 = satnogs.frame_decoder(variable_ieee802_15_4_variant_decoder_0, 1 * 1)
        self.satnogs_crc_async_0 = satnogs.crc_async(satnogs.crc.CRC32_C, False, False)
        self.satnogs_ber_calculator_0 = satnogs.ber_calculator(frame_size, nframes, 0)
        self.qtgui_time_sink_x_1 = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_1.set_update_time(0.10)
        self.qtgui_time_sink_x_1.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_1.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_1.enable_tags(True)
        self.qtgui_time_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1.enable_autoscale(False)
        self.qtgui_time_sink_x_1.enable_grid(False)
        self.qtgui_time_sink_x_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_1.enable_control_panel(False)
        self.qtgui_time_sink_x_1.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_1.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_1.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_1_win)
        self.pfb_arb_resampler_xxx_0_0_0 = pfb.arb_resampler_ccf(
            (samp_rate/(baudrate*sps_tx)),
            taps=None,
            flt_size=32,
            atten=100)
        self.pfb_arb_resampler_xxx_0_0_0.declare_sample_delay(0)
        self.pdu_pdu_to_tagged_stream_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, 'packet_len')
        self.interp_fir_filter_xxx_0 = filter.interp_fir_filter_fff(sps_tx, interp_taps)
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bf([-1, 1], 1)
        self.digital_burst_shaper_xx_0 = digital.burst_shaper_cc(([]), 200, 400, False, "packet_len")
        self.blocks_tagged_stream_multiply_length_0_0_0 = blocks.tagged_stream_multiply_length(gr.sizeof_gr_complex*1, 'packet_len', ((sps_tx * 8) ))
        self.blocks_tagged_stream_multiply_length_0_0 = blocks.tagged_stream_multiply_length(gr.sizeof_gr_complex*1, 'packet_len', ((samp_rate/(baudrate*sps_tx))))
        self.blocks_tagged_stream_align_0 = blocks.tagged_stream_align(gr.sizeof_gr_complex*1, 'packet_len')
        self.blocks_packed_to_unpacked_xx_0 = blocks.packed_to_unpacked_bb(1, gr.GR_MSB_FIRST)
        self.blocks_message_strobe_0 = blocks.message_strobe(pmt.cons(pmt.PMT_NIL,pmt.intern("TEST")), delay_ms)
        self.blocks_message_debug_0_0 = blocks.message_debug(True, gr.log_levels.info)
        self.blocks_and_const_xx_0 = blocks.and_const_bb(0)
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc(((math.pi*modulation_index) / sps_tx - (((math.pi*modulation_index) / sps_tx) * 0.1)))


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.satnogs_ber_calculator_0, 'trigger'))
        self.msg_connect((self.satnogs_ber_calculator_0, 'pdu'), (self.satnogs_frame_encoder_0, 'pdu'))
        self.msg_connect((self.satnogs_crc_async_0, 'out'), (self.pdu_pdu_to_tagged_stream_0, 'pdus'))
        self.msg_connect((self.satnogs_frame_decoder_0_0, 'out'), (self.satnogs_ber_calculator_0, 'received'))
        self.msg_connect((self.satnogs_frame_encoder_0, 'pdu'), (self.blocks_message_debug_0_0, 'print'))
        self.msg_connect((self.satnogs_frame_encoder_0, 'pdu'), (self.satnogs_crc_async_0, 'in'))
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.blocks_tagged_stream_multiply_length_0_0_0, 0))
        self.connect((self.blocks_and_const_xx_0, 0), (self.satnogs_frame_decoder_0_0, 0))
        self.connect((self.blocks_packed_to_unpacked_xx_0, 0), (self.blocks_and_const_xx_0, 0))
        self.connect((self.blocks_packed_to_unpacked_xx_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_tagged_stream_align_0, 0), (self.qtgui_time_sink_x_1, 0))
        self.connect((self.blocks_tagged_stream_multiply_length_0_0, 0), (self.blocks_tagged_stream_align_0, 0))
        self.connect((self.blocks_tagged_stream_multiply_length_0_0_0, 0), (self.digital_burst_shaper_xx_0, 0))
        self.connect((self.digital_burst_shaper_xx_0, 0), (self.pfb_arb_resampler_xxx_0_0_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.interp_fir_filter_xxx_0, 0))
        self.connect((self.interp_fir_filter_xxx_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.blocks_packed_to_unpacked_xx_0, 0))
        self.connect((self.pfb_arb_resampler_xxx_0_0_0, 0), (self.blocks_tagged_stream_multiply_length_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "fsk_ber")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_baudrate(self):
        return self.baudrate

    def set_baudrate(self, baudrate):
        self.baudrate = baudrate
        self.set_modulation_index(self.deviation / (self.baudrate / 2.0))
        self.blocks_tagged_stream_multiply_length_0_0.set_scalar(((self.samp_rate/(self.baudrate*self.sps_tx))))
        self.pfb_arb_resampler_xxx_0_0_0.set_rate((self.samp_rate/(self.baudrate*self.sps_tx)))

    def get_delay_ms(self):
        return self.delay_ms

    def set_delay_ms(self, delay_ms):
        self.delay_ms = delay_ms
        self.blocks_message_strobe_0.set_period(self.delay_ms)

    def get_frame_size(self):
        return self.frame_size

    def set_frame_size(self, frame_size):
        self.frame_size = frame_size

    def get_nframes(self):
        return self.nframes

    def set_nframes(self, nframes):
        self.nframes = nframes

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_tagged_stream_multiply_length_0_0.set_scalar(((self.samp_rate/(self.baudrate*self.sps_tx))))
        self.pfb_arb_resampler_xxx_0_0_0.set_rate((self.samp_rate/(self.baudrate*self.sps_tx)))
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate)

    def get_sps_tx(self):
        return self.sps_tx

    def set_sps_tx(self, sps_tx):
        self.sps_tx = sps_tx
        self.set_gaussian_taps(filter.firdes.gaussian(1.0, self.sps_tx, 1.0, 4*self.sps_tx))
        self.set_sq_wave((1.0, ) * self.sps_tx)
        self.analog_frequency_modulator_fc_0.set_sensitivity(((math.pi*self.modulation_index) / self.sps_tx - (((math.pi*self.modulation_index) / self.sps_tx) * 0.1)))
        self.blocks_tagged_stream_multiply_length_0_0.set_scalar(((self.samp_rate/(self.baudrate*self.sps_tx))))
        self.blocks_tagged_stream_multiply_length_0_0_0.set_scalar(((self.sps_tx * 8) ))
        self.pfb_arb_resampler_xxx_0_0_0.set_rate((self.samp_rate/(self.baudrate*self.sps_tx)))

    def get_tx_freq(self):
        return self.tx_freq

    def set_tx_freq(self, tx_freq):
        self.tx_freq = tx_freq

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain

    def get_sq_wave(self):
        return self.sq_wave

    def set_sq_wave(self, sq_wave):
        self.sq_wave = sq_wave
        self.set_interp_taps(numpy.convolve(numpy.array(self.gaussian_taps), numpy.array(self.sq_wave)))

    def get_gaussian_taps(self):
        return self.gaussian_taps

    def set_gaussian_taps(self, gaussian_taps):
        self.gaussian_taps = gaussian_taps
        self.set_interp_taps(numpy.convolve(numpy.array(self.gaussian_taps), numpy.array(self.sq_wave)))

    def get_deviation(self):
        return self.deviation

    def set_deviation(self, deviation):
        self.deviation = deviation
        self.set_modulation_index(self.deviation / (self.baudrate / 2.0))

    def get_variable_ieee802_15_4_variant_decoder_0(self):
        return self.variable_ieee802_15_4_variant_decoder_0

    def set_variable_ieee802_15_4_variant_decoder_0(self, variable_ieee802_15_4_variant_decoder_0):
        self.variable_ieee802_15_4_variant_decoder_0 = variable_ieee802_15_4_variant_decoder_0

    def get_variable_ieee802_15_4_encoder_0(self):
        return self.variable_ieee802_15_4_encoder_0

    def set_variable_ieee802_15_4_encoder_0(self, variable_ieee802_15_4_encoder_0):
        self.variable_ieee802_15_4_encoder_0 = variable_ieee802_15_4_encoder_0

    def get_modulation_index(self):
        return self.modulation_index

    def set_modulation_index(self, modulation_index):
        self.modulation_index = modulation_index
        self.analog_frequency_modulator_fc_0.set_sensitivity(((math.pi*self.modulation_index) / self.sps_tx - (((math.pi*self.modulation_index) / self.sps_tx) * 0.1)))

    def get_interp_taps(self):
        return self.interp_taps

    def set_interp_taps(self, interp_taps):
        self.interp_taps = interp_taps
        self.interp_fir_filter_xxx_0.set_taps(self.interp_taps)



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--baudrate", dest="baudrate", type=eng_float, default=eng_notation.num_to_str(float(50e3)),
        help="Set baudrate [default=%(default)r]")
    parser.add_argument(
        "--delay-ms", dest="delay_ms", type=intx, default=100,
        help="Set delay_ms [default=%(default)r]")
    parser.add_argument(
        "--frame-size", dest="frame_size", type=intx, default=252,
        help="Set The payload frame size [default=%(default)r]")
    parser.add_argument(
        "--nframes", dest="nframes", type=intx, default=1000,
        help="Set The number of frames to send [default=%(default)r]")
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=eng_float, default=eng_notation.num_to_str(float(2e6)),
        help="Set samp_rate [default=%(default)r]")
    parser.add_argument(
        "--sps-tx", dest="sps_tx", type=intx, default=16,
        help="Set sps_tx [default=%(default)r]")
    parser.add_argument(
        "--tx-freq", dest="tx_freq", type=eng_float, default=eng_notation.num_to_str(float(435e6)),
        help="Set tx_freq [default=%(default)r]")
    parser.add_argument(
        "--tx-gain", dest="tx_gain", type=eng_float, default=eng_notation.num_to_str(float(60.0)),
        help="Set tx_gain [default=%(default)r]")
    return parser


def main(top_block_cls=fsk_ber, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(baudrate=options.baudrate, delay_ms=options.delay_ms, frame_size=options.frame_size, nframes=options.nframes, samp_rate=options.samp_rate, sps_tx=options.sps_tx, tx_freq=options.tx_freq, tx_gain=options.tx_gain)

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
