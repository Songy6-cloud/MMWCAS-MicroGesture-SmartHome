# Radar microSD configuration

This folder contains the files required by the radar boot card for the continuous 28-frame capture firmware.

## Prepare the card

Back up the existing card first. For a blank FAT32 boot partition:

1. Copy `MLO` to the root of the card first.
2. Copy `AppImage` to the root of the card.
3. Copy the complete `radar_test_vector/` folder to the root without renaming or restructuring it.

The boot partition must contain:

```text
<microSD root>/
├── MLO
├── AppImage
└── radar_test_vector/
    ├── sensor_master_config.txt
    ├── cascade_fe_master_params_MIMO.txt
    ├── algo_master_params_cascade.txt
    └── cascade/
        ├── algo_configurations.txt
        ├── cascade_fe_params_mimo_master.txt
        ├── cascade_fe_params_mimo_Map2.txt
        ├── cascade_fe_params_mimo_Map4.txt
        └── cascade_fe_params_mimo_Map8.txt
```

Firmware build: `MMWCAS_MATLAB_BURST_V66_20261117`  
AppImage SHA256: `f79615cf232acb8616cb6796bcc30d6947a202f77767490264d0981fb0899f11`

## Hardware connections

Make all connections while the radar board is powered off:

1. Insert the prepared microSD card into the radar DSP board.
2. Connect the radar DSP board Ethernet port directly to the Jetson Nano Ethernet port.
3. Connect the radar DSP board debug USB port to the Windows computer running PuTTY.
4. Connect the radar and RF boards as required by the MMWCAS hardware, then connect board power.

Use these wired-network addresses:

- Radar board: `192.168.33.180`, TCP port `29172`
- Jetson Nano: `192.168.33.30/24`

## PuTTY serial settings

The debug USB normally creates several COM ports in Windows. Check **Device Manager → Ports (COM & LPT)**; the log console is usually the second of the four TI debug COM ports. The exact COM number can change.

In PuTTY select **Serial** and use:

| Setting | Value |
| --- | --- |
| Serial line | The detected COM port, for example `COM14` |
| Speed | `115200` |
| Data bits | `8` |
| Stop bits | `1` |
| Parity | `None` |
| Flow control | `None` |

Open PuTTY before powering on the radar. A correct port shows `TDA2xx SBL Boot` after power-on. If PuTTY has no output, close it and try another TI debug COM port. If there is still no output, keep PuTTY open and restart the radar power.

## Board startup

1. **Vision SDK Usecases:** select `1` for radar use cases.
2. **Radar use case:** select `a` for cascade radar processing with NetworkTx.
5. **Network Mode:** select `2` for **TCP/IP**.

The log should show that `radar_test_vector/sensor_master_config.txt` was parsed and that NetworkTx is listening on port `29172`. Do not continue with default parameters if the log reports that the configuration files are missing or invalid.

After the NetworkTx server is ready, start the Nano program with `bash run_dsp_gesture.sh`.

`LICENSE_TI.txt` and `Processor_SDK_Radar_manifest.html` document the included TI components.
