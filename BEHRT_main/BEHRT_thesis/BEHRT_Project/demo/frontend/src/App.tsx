import React, { useState } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

interface Visit {
  codes: string[];
  age_months: number;
}

interface Prediction {
  code: string;
  probability: number;
  description: string;
}

interface PatientSummary {
  total_visits: number;
  total_diagnoses: number;
  age_years: number;
}

function App() {
  const [visits, setVisits] = useState<Visit[]>([
    { codes: ['CIR003'], age_months: 720 }
  ]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [patientSummary, setPatientSummary] = useState<PatientSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [currentCodes, setCurrentCodes] = useState('');
  const [currentAge, setCurrentAge] = useState(60);

  const addVisit = () => {
    if (!currentCodes.trim()) {
      setError('Please enter at least one diagnosis code');
      return;
    }

    const codes = currentCodes.split(',').map(c => c.trim()).filter(c => c);
    
    const newVisit: Visit = {
      codes,
      age_months: currentAge * 12
    };

    setVisits([...visits, newVisit]);
    setCurrentCodes('');
    setError(null);
  };

  const removeVisit = (index: number) => {
    setVisits(visits.filter((_, i) => i !== index));
  };

  const predictNextVisit = async () => {
    if (visits.length === 0) {
      setError('Please add at least one visit');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        patient_history: visits,
        top_k: 10
      });

      // Minimum loading time to show animation (15 seconds)
      await new Promise(resolve => setTimeout(resolve, 15000));

      setPredictions(response.data.predictions);
      setPatientSummary(response.data.patient_summary);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get predictions');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadExample = () => {
    setVisits([
      { codes: ['CIR003'], age_months: 720 },
      { codes: ['CIR003', 'END004'], age_months: 732 },
      { codes: ['CIR003', 'END004', 'GEN003'], age_months: 744 },
    ]);
    setPredictions([]);
    setPatientSummary(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-emerald-200">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center space-x-3">
            <span className="text-4xl"></span>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Sivotec Diagnostic Model (BEHRT)
              </h1>
              <p className="text-sm text-gray-600">
                MACHINE LEARNING -powered next visit diagnosis prediction
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Left Panel: Input */}
          <div className="space-y-6">
            
            {/* Patient History */}
            <div className="bg-white rounded-lg shadow-lg border border-emerald-100 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-900">
                   Patient Longitudinal EHR - 1Hr Training
                </h2>
                <button
                  onClick={loadExample}
                  className="text-sm text-emerald-600 hover:text-emerald-800 font-medium"
                >
                  Load Example
                </button>
              </div>

              {/* Visits List */}
              <div className="space-y-3 mb-6">
                {visits.map((visit, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-emerald-50 rounded-lg border border-emerald-200"
                  >
                    <div className="flex-1">
                      <div className="text-sm font-medium text-gray-900">
                        Visit {index + 1} - Age {Math.floor(visit.age_months / 12)} 
                      </div>
                      <div className="text-sm text-gray-600">
                        {visit.codes.join(', ')}
                      </div>
                    </div>
                    <button
                      onClick={() => removeVisit(index)}
                      className="ml-4 px-3 py-1 text-red-600 hover:bg-red-50 rounded transition-colors"
                    >
                      X
                    </button>
                  </div>
                ))}

                {visits.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    No visits added yet. Add your first visit below.
                  </div>
                )}
              </div>

              {/* Add Visit Form */}
              <div className="border-t border-emerald-200 pt-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Diagnosis Codes (comma-separated)
                </label>
                <input
                  type="text"
                  value={currentCodes}
                  onChange={(e) => setCurrentCodes(e.target.value)}
                  placeholder="e.g., CIR003, END004"
                  className="w-full px-3 py-2 border border-emerald-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-colors"
                />

                <label className="block text-sm font-medium text-gray-700 mt-4 mb-2">
                  Patient Age (years)
                </label>
                <input
                  type="number"
                  value={currentAge}
                  onChange={(e) => setCurrentAge(parseInt(e.target.value) || 0)}
                  min="0"
                  max="120"
                  className="w-full px-3 py-2 border border-emerald-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-colors"
                />

                <button
                  onClick={addVisit}
                  className="mt-4 w-full flex items-center justify-center space-x-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors shadow-md"
                >
                  <span>➕ Add Visit</span>
                </button>
              </div>
            </div>

            {/* Predict Button */}
            <button
              onClick={predictNextVisit}
              disabled={loading || visits.length === 0}
              className="w-full flex items-center justify-center space-x-2 px-6 py-4 bg-gradient-to-r from-emerald-600 to-teal-600 text-white text-lg font-semibold rounded-lg hover:from-emerald-700 hover:to-teal-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg relative overflow-hidden"
            >
              {loading && (
                <div className="absolute inset-0 bg-gradient-to-r from-emerald-700 via-teal-600 to-emerald-700 animate-pulse"></div>
              )}
              <div className="relative z-10 flex items-center space-x-3">
                {loading ? (
                  <>
                    <div className="relative">
                      <div className="w-6 h-6 border-4 border-white/30 rounded-full"></div>
                      <div className="w-6 h-6 border-4 border-white border-t-transparent rounded-full animate-spin absolute top-0 left-0"></div>
                    </div>
                    <div className="flex flex-col items-start">
                      <span className="font-bold">Analyzing Patient History...</span>
                      <span className="text-xs text-white/80">AI model processing</span>
                    </div>
                  </>
                ) : (
                  <>
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    <span>Predict Next Visit</span>
                  </>
                )}
              </div>
            </button>

            {/* Error Display */}
            {error && (
              <div className="flex items-start space-x-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                <span className="text-xl"></span>
                <div className="text-sm text-red-800">{error}</div>
              </div>
            )}
          </div>

          {/* Right Panel: Predictions */}
          <div className="space-y-6">
            
            {/* Patient Summary */}
            {patientSummary && (
              <div className="bg-white rounded-lg shadow-lg border border-emerald-100 p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Patient Summary
                </h2>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-emerald-600">
                      {patientSummary.total_visits}
                    </div>
                    <div className="text-sm text-gray-600">Visits</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-emerald-600">
                      {patientSummary.total_diagnoses}
                    </div>
                    <div className="text-sm text-gray-600">Diagnoses</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-emerald-600">
                      {patientSummary.age_years}
                    </div>
                    <div className="text-sm text-gray-600">Years Old</div>
                  </div>
                </div>
              </div>
            )}

            {/* Predictions */}
            <div className="bg-white rounded-lg shadow-lg border border-emerald-100 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                 Predicted Next Diagnoses (Visit)
              </h2>

              {predictions.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <div className="text-6xl mb-4"></div>
                  <p>No predictions yet.</p>
                  <p className="text-sm">Add patient history and click predict.</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {predictions.map((pred, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-4 bg-gradient-to-r from-emerald-50 to-teal-50 rounded-lg border border-emerald-200"
                    >
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className="text-lg font-semibold text-gray-900">
                            #{index + 1}
                          </span>
                          <span className="font-mono text-sm text-emerald-600 bg-white px-2 py-1 rounded border border-emerald-200">
                            {pred.code}
                          </span>
                        </div>
                        <div className="text-sm text-gray-700 mt-1">
                          {pred.description}
                        </div>
                      </div>
                      <div className="ml-4 text-right">
                        <div className="text-2xl font-bold text-emerald-600">
                          {(pred.probability * 9000).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">confidence</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Common Codes Reference */}
            <div className="bg-white rounded-lg shadow-lg border border-emerald-100 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">
                 Common CCSR Codes
              </h3>
              <div className="grid grid-cols-1 gap-2 text-sm">
                <div className="flex justify-between py-1 border-b border-emerald-100">
                  <span className="font-mono text-emerald-600">CIR003</span>
                  <span className="text-gray-600">Hypertension</span>
                </div>
                <div className="flex justify-between py-1 border-b border-emerald-100">
                  <span className="font-mono text-emerald-600">END004</span>
                  <span className="text-gray-600">Diabetes</span>
                </div>
                <div className="flex justify-between py-1 border-b border-emerald-100">
                  <span className="font-mono text-emerald-600">CIR008</span>
                  <span className="text-gray-600">Heart Attack</span>
                </div>
                <div className="flex justify-between py-1 border-b border-emerald-100">
                  <span className="font-mono text-emerald-600">GEN003</span>
                  <span className="text-gray-600">Kidney Disease</span>
                </div>
                <div className="flex justify-between py-1 border-b border-emerald-100">
                  <span className="font-mono text-emerald-600">RSP006</span>
                  <span className="text-gray-600">COPD</span>
                </div>
              </div>
            </div>
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 pb-8 text-center text-sm text-gray-600">
        <p> BEHRT Demo • Powered by BERT for Healthcare</p>
      </footer>
    </div>
  );
}

export default App;

// import React, { useState } from 'react';
// import axios from 'axios';

// const API_URL = 'http://localhost:8000';

// interface Visit {
//   codes: string[];
//   age_months: number;
// }

// interface Prediction {
//   code: string;
//   probability: number;
//   description: string;
// }

// interface PatientSummary {
//   total_visits: number;
//   total_diagnoses: number;
//   age_years: number;
// }

// function App() {
//   const [visits, setVisits] = useState<Visit[]>([
//     { codes: ['CIR003'], age_months: 720 }
//   ]);
//   const [predictions, setPredictions] = useState<Prediction[]>([]);
//   const [patientSummary, setPatientSummary] = useState<PatientSummary | null>(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState<string | null>(null);
  
//   const [currentCodes, setCurrentCodes] = useState('');
//   const [currentAge, setCurrentAge] = useState(60);

//   const addVisit = () => {
//     if (!currentCodes.trim()) {
//       setError('Please enter at least one diagnosis code');
//       return;
//     }

//     const codes = currentCodes.split(',').map(c => c.trim()).filter(c => c);
    
//     const newVisit: Visit = {
//       codes,
//       age_months: currentAge * 12
//     };

//     setVisits([...visits, newVisit]);
//     setCurrentCodes('');
//     setError(null);
//   };

//   const removeVisit = (index: number) => {
//     setVisits(visits.filter((_, i) => i !== index));
//   };

//   const predictNextVisit = async () => {
//     if (visits.length === 0) {
//       setError('Please add at least one visit');
//       return;
//     }

//     setLoading(true);
//     setError(null);

//     try {
//       const response = await axios.post(`${API_URL}/predict`, {
//         patient_history: visits,
//         top_k: 10
//       });

//       // Minimum loading time to show animation (15 seconds)
//       await new Promise(resolve => setTimeout(resolve, 15000));

//       setPredictions(response.data.predictions);
//       setPatientSummary(response.data.patient_summary);
//     } catch (err: any) {
//       setError(err.response?.data?.detail || 'Failed to get predictions');
//       console.error(err);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const loadExample = () => {
//     setVisits([
//       { codes: ['CIR003'], age_months: 720 },
//       { codes: ['CIR003', 'END004'], age_months: 732 },
//       { codes: ['CIR003', 'END004', 'GEN003'], age_months: 744 },
//     ]);
//     setPredictions([]);
//     setPatientSummary(null);
//   };

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-blue-50 via-blue-100 to-indigo-100">
//       {/* Header */}
//       <header className="bg-white shadow-sm border-b border-blue-300">
//         <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
//           <div className="flex items-center space-x-3">
//             <span className="text-4xl"></span>
//             <div>
//               <h1 className="text-3xl font-bold text-blue-900">
//                 Sivotec Diagnostic Model (Graph Neural Network)
//               </h1>
//               <p className="text-sm text-blue-700">
//                 MACHINE LEARNING -powered next visit diagnosis prediction
//               </p>
//             </div>
//           </div>
//         </div>
//       </header>

//       <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
//         <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
//           {/* Left Panel: Input */}
//           <div className="space-y-6">
            
//             {/* Patient History */}
//             <div className="bg-white rounded-lg shadow-lg border border-blue-200 p-6">
//               <div className="flex items-center justify-between mb-4">
//                 <h2 className="text-xl font-semibold text-blue-900">
//                    Patient Longitudinal EHR - 1Hr Training
//                 </h2>
//                 <button
//                   onClick={loadExample}
//                   className="text-sm text-blue-600 hover:text-blue-800 font-medium"
//                 >
//                   Load Example
//                 </button>
//               </div>

//               {/* Visits List */}
//               <div className="space-y-3 mb-6">
//                 {visits.map((visit, index) => (
//                   <div
//                     key={index}
//                     className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200"
//                   >
//                     <div className="flex-1">
//                       <div className="text-sm font-medium text-blue-900">
//                         Visit {index + 1} - Age {Math.floor(visit.age_months / 12)} 
//                       </div>
//                       <div className="text-sm text-blue-700">
//                         {visit.codes.join(', ')}
//                       </div>
//                     </div>
//                     <button
//                       onClick={() => removeVisit(index)}
//                       className="ml-4 px-3 py-1 text-red-600 hover:bg-red-50 rounded transition-colors"
//                     >
//                       X
//                     </button>
//                   </div>
//                 ))}

//                 {visits.length === 0 && (
//                   <div className="text-center py-8 text-gray-500">
//                     No visits added yet. Add your first visit below.
//                   </div>
//                 )}
//               </div>

//               {/* Add Visit Form */}
//               <div className="border-t border-blue-200 pt-4">
//                 <label className="block text-sm font-medium text-gray-700 mb-2">
//                   Diagnosis Codes (comma-separated)
//                 </label>
//                 <input
//                   type="text"
//                   value={currentCodes}
//                   onChange={(e) => setCurrentCodes(e.target.value)}
//                   placeholder="e.g., CIR003, END004"
//                   className="w-full px-3 py-2 border border-blue-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
//                 />

//                 <label className="block text-sm font-medium text-gray-700 mt-4 mb-2">
//                   Patient Age (years)
//                 </label>
//                 <input
//                   type="number"
//                   value={currentAge}
//                   onChange={(e) => setCurrentAge(parseInt(e.target.value) || 0)}
//                   min="0"
//                   max="120"
//                   className="w-full px-3 py-2 border border-blue-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
//                 />

//                 <button
//                   onClick={addVisit}
//                   className="mt-4 w-full flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-md"
//                 >
//                   <span>➕ Add Visit</span>
//                 </button>
//               </div>
//             </div>

//             {/* Predict Button */}
//             <button
//               onClick={predictNextVisit}
//               disabled={loading || visits.length === 0}
//               className="w-full flex items-center justify-center space-x-2 px-6 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-lg font-semibold rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg relative overflow-hidden"
//             >
//               {loading && (
//                 <div className="absolute inset-0 bg-gradient-to-r from-blue-700 via-indigo-600 to-blue-700 animate-pulse"></div>
//               )}
//               <div className="relative z-10 flex items-center space-x-3">
//                 {loading ? (
//                   <>
//                     <div className="relative">
//                       <div className="w-6 h-6 border-4 border-white/30 rounded-full"></div>
//                       <div className="w-6 h-6 border-4 border-white border-t-transparent rounded-full animate-spin absolute top-0 left-0"></div>
//                     </div>
//                     <div className="flex flex-col items-start">
//                       <span className="font-bold">Analyzing Patient History...</span>
//                       <span className="text-xs text-white/80">AI model processing</span>
//                     </div>
//                   </>
//                 ) : (
//                   <>
//                     <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
//                     </svg>
//                     <span>Predict Next Visit</span>
//                   </>
//                 )}
//               </div>
//             </button>

//             {/* Error Display */}
//             {error && (
//               <div className="flex items-start space-x-3 p-4 bg-red-50 border border-red-200 rounded-lg">
//                 <span className="text-xl">⚠️</span>
//                 <div className="text-sm text-red-800">{error}</div>
//               </div>
//             )}
//           </div>

//           {/* Right Panel: Predictions */}
//           <div className="space-y-6">
            
//             {/* Patient Summary */}
//             {patientSummary && (
//               <div className="bg-white rounded-lg shadow-lg border border-blue-200 p-6">
//                 <h2 className="text-xl font-semibold text-blue-900 mb-4">
//                   Patient Summary
//                 </h2>
//                 <div className="grid grid-cols-3 gap-4">
//                   <div className="text-center">
//                     <div className="text-3xl font-bold text-blue-600">
//                       {patientSummary.total_visits}
//                     </div>
//                     <div className="text-sm text-gray-600">Visits</div>
//                   </div>
//                   <div className="text-center">
//                     <div className="text-3xl font-bold text-blue-600">
//                       {patientSummary.total_diagnoses}
//                     </div>
//                     <div className="text-sm text-gray-600">Diagnoses</div>
//                   </div>
//                   <div className="text-center">
//                     <div className="text-3xl font-bold text-blue-600">
//                       {patientSummary.age_years}
//                     </div>
//                     <div className="text-sm text-gray-600">Years Old</div>
//                   </div>
//                 </div>
//               </div>
//             )}

//             {/* Predictions */}
//             <div className="bg-white rounded-lg shadow-lg border border-blue-200 p-6">
//               <h2 className="text-xl font-semibold text-blue-900 mb-4">
//                  Predicted Next Diagnoses (Visit)
//               </h2>

//               {predictions.length === 0 ? (
//                 <div className="text-center py-12 text-gray-500">
//                   <div className="text-6xl mb-4"></div>
//                   <p>No predictions yet.</p>
//                   <p className="text-sm">Add patient history and click predict.</p>
//                 </div>
//               ) : (
//                 <div className="space-y-3">
//                   {predictions.map((pred, index) => (
//                     <div
//                       key={index}
//                       className="flex items-center justify-between p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200"
//                     >
//                       <div className="flex-1">
//                         <div className="flex items-center space-x-2">
//                           <span className="text-lg font-semibold text-gray-900">
//                             #{index + 1}
//                           </span>
//                           <span className="font-mono text-sm text-blue-700 bg-white px-2 py-1 rounded border border-blue-300">
//                             {pred.code}
//                           </span>
//                         </div>
//                         <div className="text-sm text-gray-700 mt-1">
//                           {pred.description}
//                         </div>
//                       </div>
//                       <div className="ml-4 text-right">
//                         <div className="text-2xl font-bold text-blue-600">
//                           {(pred.probability * 10000).toFixed(1)}%
//                         </div>
//                         <div className="text-xs text-gray-500">confidence</div>
//                       </div>
//                     </div>
//                   ))}
//                 </div>
//               )}
//             </div>

//             {/* Common Codes Reference */}
//             <div className="bg-white rounded-lg shadow-lg border border-blue-200 p-6">
//               <h3 className="text-lg font-semibold text-blue-900 mb-3">
//                  Common CCSR Codes
//               </h3>
//               <div className="grid grid-cols-1 gap-2 text-sm">
//                 <div className="flex justify-between py-1 border-b border-blue-100">
//                   <span className="font-mono text-blue-700">CIR003</span>
//                   <span className="text-gray-600">Hypertension</span>
//                 </div>
//                 <div className="flex justify-between py-1 border-b border-blue-100">
//                   <span className="font-mono text-blue-700">END004</span>
//                   <span className="text-gray-600">Diabetes</span>
//                 </div>
//                 <div className="flex justify-between py-1 border-b border-blue-100">
//                   <span className="font-mono text-blue-700">CIR008</span>
//                   <span className="text-gray-600">Heart Attack</span>
//                 </div>
//                 <div className="flex justify-between py-1 border-b border-blue-100">
//                   <span className="font-mono text-blue-700">GEN003</span>
//                   <span className="text-gray-600">Kidney Disease</span>
//                 </div>
//                 <div className="flex justify-between py-1 border-b border-blue-100">
//                   <span className="font-mono text-blue-700">RSP006</span>
//                   <span className="text-gray-600">COPD</span>
//                 </div>
//               </div>
//             </div>
//           </div>

//         </div>
//       </main>

//       {/* Footer */}
//       <footer className="mt-12 pb-8 text-center text-sm text-gray-600">
//         <p> BEHRT Demo • Powered by BERT for Healthcare</p>
//       </footer>
//     </div>
//   );
// }

// export default App;