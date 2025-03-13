#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: SatNOGS-COMMS FSK transceiver
# GNU Radio version: 3.10.11.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import analog
from gnuradio import blocks
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
from gnuradio import zeromq
from gnuradio.filter import pfb
import gnuradio.satnogs as satnogs
import math
import numpy
import osmosdr
import time
import threading



class fsk(gr.top_block, Qt.QWidget):

    def __init__(self, baudrate=50e3, freq=2e6, rx_freq=435e6, rx_gain=30.0, samp_rate=2e6, sps_rx=4, sps_tx=16, tx_freq=435e6, tx_gain=60.0, zmq_pub_uri='tcp://127.0.0.1:55000', zmq_sub_uri='tcp://127.0.0.1:55001'):
        gr.top_block.__init__(self, "SatNOGS-COMMS FSK transceiver", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("SatNOGS-COMMS FSK transceiver")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "fsk")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Parameters
        ##################################################
        self.baudrate = baudrate
        self.freq = freq
        self.rx_freq = rx_freq
        self.rx_gain = rx_gain
        self.samp_rate = samp_rate
        self.sps_rx = sps_rx
        self.sps_tx = sps_tx
        self.tx_freq = tx_freq
        self.tx_gain = tx_gain
        self.zmq_pub_uri = zmq_pub_uri
        self.zmq_sub_uri = zmq_sub_uri

        ##################################################
        # Variables
        ##################################################
        self.sq_wave = sq_wave = (1.0, ) * sps_tx
        self.gaussian_taps = gaussian_taps = filter.firdes.gaussian(1.0, sps_tx, 1.0, 4*sps_tx)
        self.deviation = deviation = 25e3
        self.variable_ieee802_15_4_variant_decoder_0 = variable_ieee802_15_4_variant_decoder_0 = satnogs.ieee802_15_4_variant_decoder([0b01010101]*12, 0, [0x1A, 0xCF, 0xFC, 0x1D], 4, satnogs.crc.CRC32_C, satnogs.whitening.make_ccsds(True), False, (448-4), True, False)
        self.variable_ieee802_15_4_encoder_0 = variable_ieee802_15_4_encoder_0 = satnogs.ieee802_15_4_encoder(0b01010101, 12, [0x1A, 0xCF, 0xFC, 0x1D], satnogs.crc.CRC32_C, satnogs.whitening.make_none(), False, False)
        self.modulation_index = modulation_index = deviation / (baudrate / 2.0)
        self.interp_taps = interp_taps = numpy.convolve(numpy.array(gaussian_taps), numpy.array(sq_wave))

        ##################################################
        # Blocks
        ##################################################

        self.zeromq_sub_msg_source_0 = zeromq.sub_msg_source(zmq_sub_uri, 100, False)
        self.satnogs_frame_encoder_0 = satnogs.frame_encoder(variable_ieee802_15_4_encoder_0)
        self.satnogs_crc_async_0 = satnogs.crc_async(satnogs.crc.CRC16_CMS, False, False)
        self.pfb_arb_resampler_xxx_0_0_0 = pfb.arb_resampler_ccf(
            (samp_rate/(baudrate*sps_tx)),
            taps=None,
            flt_size=32,
            atten=100)
        self.pfb_arb_resampler_xxx_0_0_0.declare_sample_delay(0)
        self.pdu_pdu_to_tagged_stream_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, 'packet_len')
        self.osmosdr_sink_0 = osmosdr.sink(
            args="numchan=" + str(1) + " " + "bladerf"
        )
        self.osmosdr_sink_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_sink_0.set_sample_rate(samp_rate)
        self.osmosdr_sink_0.set_center_freq(401e6, 0)
        self.osmosdr_sink_0.set_freq_corr(0, 0)
        self.osmosdr_sink_0.set_gain(10, 0)
        self.osmosdr_sink_0.set_if_gain(20, 0)
        self.osmosdr_sink_0.set_bb_gain(20, 0)
        self.osmosdr_sink_0.set_antenna('TX1', 0)
        self.osmosdr_sink_0.set_bandwidth(0, 0)
        self.interp_fir_filter_xxx_0 = filter.interp_fir_filter_fff(sps_tx, interp_taps)
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bf([-1, 1], 1)
        self.digital_burst_shaper_xx_0 = digital.burst_shaper_cc(([]), 200, 400, False, "packet_len")
        self.blocks_tagged_stream_multiply_length_0_0_0 = blocks.tagged_stream_multiply_length(gr.sizeof_gr_complex*1, 'packet_len', ((sps_tx * 8) ))
        self.blocks_tagged_stream_multiply_length_0_0 = blocks.tagged_stream_multiply_length(gr.sizeof_gr_complex*1, 'packet_len', ((samp_rate/(baudrate*sps_tx))))
        self.blocks_tagged_stream_align_0 = blocks.tagged_stream_align(gr.sizeof_gr_complex*1, 'packet_len')
        self.blocks_packed_to_unpacked_xx_0 = blocks.packed_to_unpacked_bb(1, gr.GR_MSB_FIRST)
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc(((math.pi*modulation_index) / sps_tx - (((math.pi*modulation_index) / sps_tx) * 0.1)))


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.satnogs_crc_async_0, 'out'), (self.pdu_pdu_to_tagged_stream_0, 'pdus'))
        self.msg_connect((self.satnogs_frame_encoder_0, 'pdu'), (self.satnogs_crc_async_0, 'in'))
        self.msg_connect((self.zeromq_sub_msg_source_0, 'out'), (self.satnogs_frame_encoder_0, 'pdu'))
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.blocks_tagged_stream_multiply_length_0_0_0, 0))
        self.connect((self.blocks_packed_to_unpacked_xx_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_tagged_stream_align_0, 0), (self.osmosdr_sink_0, 0))
        self.connect((self.blocks_tagged_stream_multiply_length_0_0, 0), (self.blocks_tagged_stream_align_0, 0))
        self.connect((self.blocks_tagged_stream_multiply_length_0_0_0, 0), (self.digital_burst_shaper_xx_0, 0))
        self.connect((self.digital_burst_shaper_xx_0, 0), (self.pfb_arb_resampler_xxx_0_0_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.interp_fir_filter_xxx_0, 0))
        self.connect((self.interp_fir_filter_xxx_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.blocks_packed_to_unpacked_xx_0, 0))
        self.connect((self.pfb_arb_resampler_xxx_0_0_0, 0), (self.blocks_tagged_stream_multiply_length_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "fsk")
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

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq

    def get_rx_freq(self):
        return self.rx_freq

    def set_rx_freq(self, rx_freq):
        self.rx_freq = rx_freq

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_tagged_stream_multiply_length_0_0.set_scalar(((self.samp_rate/(self.baudrate*self.sps_tx))))
        self.osmosdr_sink_0.set_sample_rate(self.samp_rate)
        self.pfb_arb_resampler_xxx_0_0_0.set_rate((self.samp_rate/(self.baudrate*self.sps_tx)))

    def get_sps_rx(self):
        return self.sps_rx

    def set_sps_rx(self, sps_rx):
        self.sps_rx = sps_rx

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

    def get_zmq_pub_uri(self):
        return self.zmq_pub_uri

    def set_zmq_pub_uri(self, zmq_pub_uri):
        self.zmq_pub_uri = zmq_pub_uri

    def get_zmq_sub_uri(self):
        return self.zmq_sub_uri

    def set_zmq_sub_uri(self, zmq_sub_uri):
        self.zmq_sub_uri = zmq_sub_uri

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
        "--rx-freq", dest="rx_freq", type=eng_float, default=eng_notation.num_to_str(float(435e6)),
        help="Set rx_freq [default=%(default)r]")
    parser.add_argument(
        "--rx-gain", dest="rx_gain", type=eng_float, default=eng_notation.num_to_str(float(30.0)),
        help="Set rx_gain [default=%(default)r]")
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=eng_float, default=eng_notation.num_to_str(float(2e6)),
        help="Set samp_rate [default=%(default)r]")
    parser.add_argument(
        "--sps-rx", dest="sps_rx", type=intx, default=4,
        help="Set sps_rx [default=%(default)r]")
    parser.add_argument(
        "--sps-tx", dest="sps_tx", type=intx, default=16,
        help="Set sps_tx [default=%(default)r]")
    parser.add_argument(
        "--tx-freq", dest="tx_freq", type=eng_float, default=eng_notation.num_to_str(float(435e6)),
        help="Set tx_freq [default=%(default)r]")
    parser.add_argument(
        "--tx-gain", dest="tx_gain", type=eng_float, default=eng_notation.num_to_str(float(60.0)),
        help="Set tx_gain [default=%(default)r]")
    parser.add_argument(
        "--zmq-pub-uri", dest="zmq_pub_uri", type=str, default='tcp://127.0.0.1:55000',
        help="Set zmq_pub_uri [default=%(default)r]")
    parser.add_argument(
        "--zmq-sub-uri", dest="zmq_sub_uri", type=str, default='tcp://127.0.0.1:55001',
        help="Set zmq_sub_uri [default=%(default)r]")
    return parser


def main(top_block_cls=fsk, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(baudrate=options.baudrate, rx_freq=options.rx_freq, rx_gain=options.rx_gain, samp_rate=options.samp_rate, sps_rx=options.sps_rx, sps_tx=options.sps_tx, tx_freq=options.tx_freq, tx_gain=options.tx_gain, zmq_pub_uri=options.zmq_pub_uri, zmq_sub_uri=options.zmq_sub_uri)

    tb.start()
    tb.flowgraph_started.set()

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
