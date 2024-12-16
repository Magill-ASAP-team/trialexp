import { useState } from 'react'

import '@mantine/core/styles.css'
import { createTheme, MantineProvider } from '@mantine/core'
import { Button, Select } from '@mantine/core'
import { Flex } from '@mantine/core'

function App() {
  const [count, setCount] = useState(0)
  const [cohort, setCohort] = useState(['August','April'])
  const [animalID, setAnimalID] = useState(['A','B','C'])
  const [sessionID, setSessionID] = useState(['1','2','3'])  

  return (
    <>
      <MantineProvider>
        <Flex justify="center" align="center" gap="md">
          <Button> Hello, mantine! </Button>

          <Select data={cohort} placeholder="Select your favorite framework" />
          <Select data={animalID} placeholder="Select your favorite framework" />
          <Select data={sessionID} placeholder="Select your favorite framework" />

        </Flex>
      </MantineProvider>


    </>
  )
}

export default App
