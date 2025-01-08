import { useEffect, useState } from 'react'

import '@mantine/core/styles.css'
import { Center, MantineProvider } from '@mantine/core'
import { Button, Select, Flex, Slider, Text, Stack, Group } from '@mantine/core'
import axios from 'axios'
import Plot from 'react-plotly.js'

function App() {
  const [cohortList, setCohortList] = useState(['August', 'April'])
  const [animalIDList, setAnimalIDList] = useState(['A', 'B', 'C'])
  const [sessionIDList, setSessionIDList] = useState(['1', '2', '3'])
  const [depthShift, setDepthShift] = useState(0)
  const [cohort, setCohort] = useState<string | null>(null)
  const [animalID, setAnimalID] = useState<string | null>(null)
  const [sessionID, setSessionID] = useState<string | null>(null)
  const [plotData, setPlotData] = useState([])
  const [cellMetricsData, setCellMetricsData] = useState([])


  useEffect(() => {
    // a async function to fetch data
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/cohorts')
        if (!response.ok) {
          throw new Error('Server error');
        }
        const result = await response.json();
        setCohortList(result['cohorts'])
      } catch (err) {
        console.log('Error:', err);
      }
    }

    fetchData();
  }, []);

  useEffect(() => {
    // a async function to fetch data
    const fetchData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/animal_id?cohort=${cohort}`)
        if (!response.ok) {
          throw new Error('Server error');
        }
        const result = await response.json();
        setAnimalID(null);
        setAnimalIDList(result['animal_id']);
      } catch (err) {
        console.log('Error:', err);
      }
    }

    fetchData();
  }, [cohort]);


  useEffect(() => {
    axios.get('http://localhost:8000/sessions',
      { params: { cohort: cohort, animal_id: animalID } })
      .then((response) => {
        setSessionIDList(null);
        setSessionIDList(response.data['session_id']);
      })
      .catch((error) => {
        console.log(error);
      })
      .finally(() => { });

  }, [cohort, animalID]);

  const fetchPlotData = () => {
    if (sessionID) {
      axios.get('http://localhost:8000/trajectory',
        { params: { session_id: sessionID, shift: depthShift } })
        .then((response) => {
          setPlotData(response.data);
        })
        .catch((error) => {
          console.log(error);
        });
    }
  };



  useEffect(() => {
    fetchPlotData();
  }, [sessionID, depthShift]);

  //Get firing rate data
  useEffect(() => {
    if (sessionID) {
      axios.get('http://localhost:8000/cell_metrics',
        { params: { session_id: sessionID } })
        .then((response) => {
          setCellMetricsData(response.data);
        })
        .catch((error) => {
          console.log(error);
        });
    }
  }, [sessionID])

  const plotTraces = plotData.map((region: any) => ({
    //coordinates starts counting from the tip
    x: ['Brain Regions'],
    y: [region.depth_start - region.depth_end],
    base: [region.depth_end],
    name: region.acronym,
    type: 'bar',
    text: region.name,
    textposition: 'inside'
  }));

  const frPlotTraces = cellMetricsData && Object.keys(cellMetricsData).length > 0 ? [{
    x: cellMetricsData['mean'],
    y: cellMetricsData['ks_chan_pos_y'],
    type: 'bar',
    orientation: 'h'
  }] : [];

  const cellCountTraces = cellMetricsData && Object.keys(cellMetricsData).length > 0 ? [{
    x: cellMetricsData['count'],
    y: cellMetricsData['ks_chan_pos_y'],
    type: 'bar',
    orientation: 'h'
  }] : [];

  return (
    <>
      <MantineProvider>

        <Stack>
          <Flex justify="center" align="flex-end" gap="md">
            <Select data={cohortList} placeholder="Cohort"
              label='Cohort'
              value={cohort} onChange={setCohort} searchable />
            <Select data={animalIDList} placeholder="Animal" label='Animal'
              value={animalID} onChange={setAnimalID} searchable />
            <Select data={sessionIDList} placeholder="Session" label='Session'
              value={sessionID} onChange={setSessionID} searchable />
            <Button onClick={fetchPlotData}> Align probe locations</Button>
          </Flex>

          <Center>
            <Stack >
              <Text> Probe depth shift</Text>
              <Group>
                <Slider
                  value={depthShift}
                  onChange={setDepthShift}
                  min={-2000}
                  max={2000}
                  labelAlwaysOn
                  style={{ width: 600 }} />
                <Button onClick={() => setDepthShift(0)}>Reset</Button>
                <Button>Save</Button>
              </Group>

            </Stack>
          </Center>

        </Stack>

        <Center>
            <Group>
            <Plot
              data={plotTraces}
              layout={{
              title: plotData[0]['track_date'] ? `Mapped trajectory<br> ${plotData[0]['track_date']}` : 'Mapped trajectory',
              barmode: 'stack',
              xaxis: { title: 'Brain Regions' },
              yaxis: { title: 'Distance from tip (µm)', range: [0, 4000] },
              legend: { traceorder: 'normal' },
              width: 400,
              height: 1000,
              base: 1000,
              }}
            />

            <Plot
              data={frPlotTraces}
              layout={{
              title: 'Firing Rate by Position (experiment)',
              xaxis: { title: 'Firing Rate (Hz)' },
              yaxis: { title: 'Distance from tip (µm)', range: [0, 4000] },
              width: 400,
              height: 1000,
              orientation: 'h'
              }}
            />
            <Plot
              data={cellCountTraces}
              layout={{
              title: 'Cell count',
              xaxis: { title: 'Cell count' },
              yaxis: { title: 'Distance from tip (µm)', range: [0, 4000] },
              width: 400,
              height: 1000,
              orientation: 'h'
              }}
            />
            </Group>
        </Center>



      </MantineProvider>
    </>
  )
}

export default App
