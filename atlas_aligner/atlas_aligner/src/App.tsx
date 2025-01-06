import { useEffect, useState } from 'react'

import '@mantine/core/styles.css'
import { Center, MantineProvider } from '@mantine/core'
import { Button, Select, Flex, Slider, Text, Stack } from '@mantine/core'
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

  const plotTraces = plotData.map((region: any) => ({
    x: ['Brain Regions'],
    y: [region.depth_start - region.depth_end],
    name: region.acronym,
    type: 'bar',
    text: region.acronym,
    textposition: 'inside'
  }));

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
              <Slider
                value={depthShift}
                onChange={setDepthShift}
                min={-1000}
                max={1000}
                labelAlwaysOn
                style={{ width: 600 }} />
            </Stack>
          </Center>

        </Stack>



        <Plot
          data={plotTraces}
          layout={{
            title: 'Mapped trajectory',
            barmode: 'stack',
            xaxis: { title: 'Brain Regions' },
            yaxis: { title: 'Depth', autorange: 'reversed' },
            legend: { traceorder: 'normal' }, // Normal order of the legend
            width: 400, // Set the width of the plot here
            height: 1000, // Set the height of the plot here
          }}
        />

      </MantineProvider>
    </>
  )
}

export default App
