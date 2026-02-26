import React from 'react';
import './LabTests.css';

const labTests = [
  { id: 1, name: 'Health Checkups', image: 'LabTests1.png' },
  { id: 2, name: 'Diabetes Test', image: 'LabTests2.png' },
  { id: 3, name: 'Thyroid Test', image: 'LabTests3.png' },
  { id: 4, name: 'Women Care', image: 'LabTests4.png' },
  { id: 5, name: 'Fever and Injections', image: 'LabTests5.png' },
  { id: 6, name: 'Lifestyle', image: 'LabTests6.png' },
];

export default function LabTests() {
  return (
    <div className="lab-tests-container">
      <h2 className="lab-tests-title">Available Lab Tests</h2>
      <div className="lab-tests-wrapper">
        {labTests.map(test => (
          <div className="lab-test-box" key={test.id}>
            <img src={test.image} alt={test.name} className="lab-test-image" />
            <p className="lab-test-name">{test.name}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
