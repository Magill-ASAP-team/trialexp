import { useEffect, useState } from 'react'

import '@mantine/core/styles.css'
import { createTheme, MantineProvider } from '@mantine/core'
import { Button, Select } from '@mantine/core'
import { Flex } from '@mantine/core'
import axios from 'axios'

function App() {
  const [count, setCount] = useState(0)
  const [cohortList, setCohortList] = useState(['August', 'April'])
  const [animalIDList, setAnimalIDList] = useState(['A', 'B', 'C'])
  const [sessionIDList, setSessionIDList] = useState(['1', '2', '3'])
  const [cohort, setCohort] = useState<string | null>(null)
  const [animalID, setAnimalID] = useState<string | null>(null)
  const [sessionID, setSessionID] = useState<string | null>(null)

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
      .finally(() => {});
      
  }, [cohort, animalID]);



  return (
    <>
      <MantineProvider>
        <Flex justify="center" align="flex-end" gap="md">
          <Select data={cohortList} placeholder="Cohort"
            label='Cohort'
            value={cohort} onChange={setCohort} searchable />
          <Select data={animalIDList} placeholder="Animal" label='Animal'
            value={animalID} onChange={setAnimalID} searchable/>
          <Select data={sessionIDList} placeholder="Session" label='Session'
            value={sessionID} onChange={setSessionID} searchable />
          <Button> Align probe locations</Button>
        </Flex>
      </MantineProvider>


    </>
  )
}

export default App
