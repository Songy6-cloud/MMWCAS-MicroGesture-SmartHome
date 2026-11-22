"""TI Radar SDK 3.8 cascade object-detect TCP metadata (Python 3.6).

Wire definitions: networkCtrl_if.h and radarDspCascadeMimo_if.h.
This is NOT the TI mmWave demo UART protocol or the ADC capture server.
"""
import math
import struct

NET_HEADER = struct.Struct('<9I')
RADAR_HEADER = struct.Struct('<8s5I')
SECTION = struct.Struct('<2I')
POINT = struct.Struct('<4h9f')
MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MAX_BYTES = 4 * 1024 * 1024


def read_exact(sock, count):
    result = bytearray()
    while len(result) < count:
        data = sock.recv(count - len(result))
        if not data:
            raise EOFError('DSP closed the TCP stream')
        result.extend(data)
    return bytes(result)


def parse_radar(payload, include_rd=False):
    if len(payload) < RADAR_HEADER.size:
        raise ValueError('Truncated radar header')
    magic, frame, dsp, sections, total, objects = RADAR_HEADER.unpack_from(payload)
    if magic != MAGIC or not RADAR_HEADER.size <= total <= len(payload):
        raise ValueError('Invalid radar magic or total size')
    if sections > 32 or objects > 4096:
        raise ValueError('Invalid section/object count')
    offset = RADAR_HEADER.size
    points, polar, bins, system, has_points = [], [], [], None, False
    rd = None
    burst = None
    heat = None
    for _ in range(sections):
        if offset + SECTION.size > total:
            raise ValueError('Truncated section header')
        kind, length = SECTION.unpack_from(payload, offset)
        offset += SECTION.size
        end = offset + length
        if end > total:
            raise ValueError('Section extends beyond radar packet')
        if kind == 1:
            if has_points or length != objects * POINT.size:
                raise ValueError('Unexpected point layout/count; use normal SDK point output')
            has_points = True
            for pos in range(offset, end, POINT.size):
                fields = POINT.unpack_from(payload, pos)
                # SDK rangeInd is zero-based; dopplerInd is already signed.
                bins.append((fields[0], fields[1]))
                r, v, az, el, snr = fields[4:9]
                if not all(math.isfinite(x) for x in fields[4:]) or r < 0:
                    raise ValueError('Invalid point values')
                # Same convention as TI radar_cascade_demo.m; no extra mounting rotation.
                points.append((r * math.cos(el) * math.sin(az),
                               r * math.cos(el) * math.cos(az),
                               -r * math.sin(el)))
                polar.append((r, v, az, el, snr))
        elif kind == 0x4232:
            if burst is not None or length!=64:raise ValueError('Invalid burst metadata')
            v=struct.unpack_from('<16I',payload,offset)
            if v[0]!=1 or v[2]>4 or v[3]>28 or v[4]>28 or v[12]!=28 or v[13]!=70000 or v[14]!=1:
                raise ValueError('Unsupported burst protocol')
            burst=dict(version=v[0],session=v[1],phase=v[2],captured=v[3],index=v[4],error=v[5],
                       first_us=v[6]+(v[7]<<32),last_us=v[8]+(v[9]<<32),sample_us=v[10]+(v[11]<<32))
        elif kind == 0x5244:
            if rd is not None or length < 72:
                raise ValueError('Invalid RD diagnostic section')
            values = struct.unpack_from('<16I2f', payload, offset)
            nr, nd = values[1:3]
            if values[0] not in (1, 2) or not 1 <= nr <= 256 or not 2 <= nd <= 256 or nd % 2 or length != 72+nr*nd:
                raise ValueError('Unsupported RD diagnostic version/dimensions')
            if not all(math.isfinite(v) for v in values[16:]):
                raise ValueError('Nonfinite CFAR settings')
            names=('version','range_bins','doppler_bins','min_bin','max_bin','min_area',
                   'raw_cells','in_range_cells','components','valid_components','selected_cells',
                   'aoa_input_cells','selected_min_bin','selected_max_bin','cap_dropped','selection_enabled','k0','kd')
            rd=dict(zip(names,values))
            rd['pipeline']='matlab_v5' if values[0]==2 else 'sdk_rd_v2'
            if include_rd:
                mask=payload[offset+72:end]
                if any(x not in (0,1,3,7) for x in mask):
                    raise ValueError('Invalid RD mask flags')
                rd['cells']=[[i % nr+1, i//nr if i//nr<nd//2 else i//nr-nd, flag]
                             for i,flag in enumerate(mask) if flag]
        elif kind == 4 and include_rd:
            if heat is not None or length % 4 or length > 256*256*4:
                raise ValueError('Invalid RD heatmap size')
            heat=struct.unpack_from('<%df' % (length//4),payload,offset)
            if not all(math.isfinite(v) and v >= 0 for v in heat):
                raise ValueError('Invalid RD linear power values')
        elif kind == 5:
            if system is not None or length != 20:
                raise ValueError('Unexpected system-information layout')
            values = struct.unpack_from('<ffHH5B3x', payload, offset)
            system = dict(zip(('range_resolution_m', 'doppler_resolution_m_s',
                               'range_bins', 'doppler_bins', 'sensors', 'tx_per_sensor',
                               'rx_per_sensor', 'azimuth_tx', 'elevation_tx'), values))
        offset = end
    if offset != total:
        raise ValueError('Radar section sizes do not match total size')
    if not has_points:
        raise ValueError('Point-cloud section disabled; enable pointCloud in DSP GUI output')
    if rd and system and (rd['range_bins'] != system['range_bins'] or rd['doppler_bins'] != system['doppler_bins']):
        raise ValueError('RD/system dimensions disagree')
    if heat is not None:
        if not system or len(heat) != system['range_bins']*system['doppler_bins']:
            raise ValueError('RD heatmap/system dimensions disagree')
        if rd is not None:
            rd['power'] = heat
    return {'frame_id': frame, 'dsp_id': dsp, 'xyz': points,
            'polar': polar, 'bins': bins, 'system': system, 'rd':rd,'burst':burst}


def receive_frame(sock, include_rd=False):
    buffers, magic, kind, channel, size, _, _, _, _ = NET_HEADER.unpack(read_exact(sock, NET_HEADER.size))
    if magic != 0xABCD4321 or buffers != 1 or kind != 1:
        raise ValueError('Expected one TI cascade TCP metadata buffer; select TCP/IP on DSP')
    if not RADAR_HEADER.size <= size <= MAX_BYTES:
        raise ValueError('Invalid TCP payload size')
    frame = parse_radar(read_exact(sock, size), include_rd=include_rd)
    frame['channel'] = channel
    return frame
