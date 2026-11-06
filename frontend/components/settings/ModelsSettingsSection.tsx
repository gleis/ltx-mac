import { useState } from 'react'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { SettingsSubsection } from './SettingsContent'

import { BaseModelSection } from './BaseModelSection'
import { ImageGenModelSection } from './ImageGenModelSection'
import { LoraSettingsSection } from './LoraSettingsSection'
import { ModelsCatalogSearch } from './ModelsCatalogSearch'
import { ModelsCatalogSearchEmpty } from './ModelsCatalogSearchEmpty'
import { EncodingEnhanceModelsSection } from './EncodingEnhanceModelsSection'
import { AssistantModelsSection } from './AssistantModelsSection'
import { ModelsManagementSection } from './ModelsManagementSection'
import { SettingsAnchorSection } from './SettingsAnchorSection'
import styles from './SettingsPageLayout.module.scss'

export function ModelsSettingsSection({
  sectionActive,
}: {
  sectionActive: boolean
}) {
  const { settings, updateSettings, localViable, cudaAvailable } = useAppSettings()
  const [searchQuery, setSearchQuery] = useState('')
  const [baseHasMatches, setBaseHasMatches] = useState(true)
  const [loraHasMatches, setLoraHasMatches] = useState(true)
  const [imageHasMatches, setImageHasMatches] = useState(true)
  const [encodingHasMatches, setEncodingHasMatches] = useState(true)
  const [assistantHasMatches, setAssistantHasMatches] = useState(true)
  const normalizedSearch = searchQuery.trim().toLowerCase()
  const isSearching = normalizedSearch.length > 0
  const anySearchMatches =
    !isSearching ||
    baseHasMatches ||
    loraHasMatches ||
    imageHasMatches ||
    encodingHasMatches ||
    assistantHasMatches

  const handleSearchChange = (value: string) => {
    setSearchQuery(value)
    setBaseHasMatches(true)
    setLoraHasMatches(true)
    setImageHasMatches(true)
    setEncodingHasMatches(true)
    setAssistantHasMatches(true)
  }

  return (
    <SettingsAnchorSection
      title="Models"
      headerTrailing={<ModelsCatalogSearch value={searchQuery} onChange={handleSearchChange} />}
    >
      {!isSearching ? <ModelsManagementSection /> : null}
      {!isSearching && localViable && !cudaAvailable && window.electronAPI.platform === 'darwin' ? (
        <SettingsSubsection title="Local MLX generation" description="Speed mode applies to text-to-video. Image-to-video uses guided quality mode.">
          <label htmlFor="mlx-speed">Speed mode </label>
          <select id="mlx-speed" value={settings.localMlxSpeedMode} onChange={event => {
            const value = event.target.value
            if (value === 'quality' || value === 'boost' || value === 'turbo') updateSettings({ localMlxSpeedMode: value })
          }}>
            <option value="quality">Quality</option><option value="boost">Boost</option><option value="turbo">Turbo</option>
          </select>
        </SettingsSubsection>
      ) : null}
      <div
        className={styles.modelsCatalogStack}
        hidden={isSearching && !anySearchMatches}
      >
        <BaseModelSection
          searchQuery={normalizedSearch}
          onSearchHasMatches={setBaseHasMatches}
        />
        <LoraSettingsSection
          searchQuery={normalizedSearch}
          onSearchHasMatches={setLoraHasMatches}
        />
        <ImageGenModelSection
          searchQuery={normalizedSearch}
          onSearchHasMatches={setImageHasMatches}
        />
        <EncodingEnhanceModelsSection
          active={sectionActive}
          searchQuery={normalizedSearch}
          onSearchHasMatches={setEncodingHasMatches}
        />
        <AssistantModelsSection
          active={sectionActive}
          searchQuery={normalizedSearch}
          onSearchHasMatches={setAssistantHasMatches}
        />
      </div>
      {isSearching && !anySearchMatches ? (
        <div className={styles.modelsSearchEmptyHost}>
          <ModelsCatalogSearchEmpty query={searchQuery.trim()} />
        </div>
      ) : null}
    </SettingsAnchorSection>
  )
}
