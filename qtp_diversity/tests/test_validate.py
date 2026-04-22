# -----------------------------------------------------------------------------
# Copyright (c) 2014--, The Qiita Development Team.
#
# Distributed under the terms of the BSD 3-clause License.
#
# The full license is in the file LICENSE, distributed with this software.
# -----------------------------------------------------------------------------

from unittest import main
from tempfile import mkdtemp, mkstemp
from os.path import exists, isdir, join
from os import remove, close
from shutil import rmtree
from json import dumps

from skbio.stats.distance import randdm
from skbio import OrdinationResults
from qiita_client import ArtifactInfo
from qiita_client.testing import PluginTestCase
import pandas as pd
import numpy as np

from qtp_diversity import plugin
from qtp_diversity.validate import (
    _validate_distance_matrix, _validate_ordination_results,
    _validate_alpha_vector, _validate_feature_data, validate,
    _validate_sample_data)


class ValidateTests(PluginTestCase):
    def setUp(self):
        self.out_dir = mkdtemp(prefix=self.base_data_dir)
        self._clean_up_files = [self.out_dir]
        self.metadata = {
            '1.SKM4.640180': {'col': "doesn't really matters"},
            '1.SKB8.640193': {'col': "doesn't really matters"},
            '1.SKD8.640184': {'col': "doesn't really matters"},
            '1.SKM9.640192': {'col': "doesn't really matters"},
            '1.SKB7.640196': {'col': "doesn't really matters"}}

        plugin('https://localhost:8383', 'register', 'ignored')

    def tearDown(self):
        for fp in self._clean_up_files:
            if exists(fp):
                if isdir(fp):
                    rmtree(fp)
                else:
                    remove(fp)

    def _create_distance_matrix(self, sample_ids):
        dm = randdm(len(sample_ids), sample_ids)
        fd, fp = mkstemp(suffix='.txt', dir=self.out_dir)
        close(fd)
        dm.write(fp)
        return fp

    def _create_ordination_results(self, sample_ids):
        eigvals = [0.51236726, 0.30071909, 0.26791207, 0.20898868]
        proportion_explained = [0.2675738328, 0.157044696, 0.1399118638,
                                0.1091402725]
        axis_labels = ['PC1', 'PC2', 'PC3', 'PC4']
        samples = [[-2.584, 1.739, 3.828, -1.944],
                   [-2.710, -1.859, -8.648, 1.180],
                   [2.350, 9.625, -3.457, -3.208],
                   [2.614, -1.114, 1.476, 2.908],
                   [2.850, -1.925, 6.232, 1.381]]
        ord_res = OrdinationResults(
            short_method_name='PCoA',
            long_method_name='Principal Coordinate Analysis',
            eigvals=pd.Series(eigvals, index=axis_labels),
            samples=pd.DataFrame(np.asarray(samples), index=sample_ids,
                                 columns=axis_labels),
            proportion_explained=pd.Series(proportion_explained,
                                           index=axis_labels))
        fd, fp = mkstemp(suffix='.txt', dir=self.out_dir)
        close(fd)
        ord_res.write(fp)
        return fp

    def _create_alpha_vector(self, sample_ids):
        fd, fp = mkstemp(suffix='.txt', dir=self.out_dir)
        close(fd)
        with open(fp, 'w') as f:
            f.write("\tobserved_otus\n")
            for s_id in sample_ids:
                f.write("%s\t%d\n" % (s_id, np.random.randint(1, 200)))

        return fp

    def _create_job(self, a_type, files, analysis):
        parameters = {'template': None,
                      'files': dumps(files),
                      'artifact_type': a_type,
                      'analysis': analysis}
        data = {'command': dumps(['Diversity types', '2023.02', 'Validate']),
                'parameters': dumps(parameters),
                'status': 'running'}
        job_id = self.qclient.post(
            '/apitest/processing_job/', data=data)['job']
        return job_id, parameters

    def test_validate_distance_matrix(self):
        # Create a distance matrix
        sample_ids = ['1.SKM4.640180', '1.SKB8.640193', '1.SKD8.640184',
                      '1.SKM9.640192', '1.SKB7.640196']
        dm_fp = self._create_distance_matrix(sample_ids)

        # Test success
        obs_success, obs_ainfo, obs_error = _validate_distance_matrix(
            {'plain_text': [dm_fp]}, self.metadata, self.out_dir)
        self.assertTrue(obs_success)
        exp_ainfo = [ArtifactInfo(None, "distance_matrix",
                                  [(dm_fp, 'plain_text')])]
        self.assertEqual(obs_ainfo, exp_ainfo)
        self.assertEqual(obs_error, "")

        # Test failure
        sample_ids = ['1.SKM4.640180', '1.SKB8.640193', '1.SKD8.640184',
                      '1.SKM9.640192', 'NotASample']
        dm_fp = self._create_distance_matrix(sample_ids)
        obs_success, obs_ainfo, obs_error = _validate_distance_matrix(
            {'plain_text': [dm_fp]}, self.metadata, self.out_dir)
        self.assertFalse(obs_success)
        self.assertIsNone(obs_ainfo)
        self.assertEqual(obs_error, "The distance matrix contain samples not "
                                    "present in the metadata")

    def test_validate_ordination_results(self):
        # Create the ordination results
        sample_ids = ['1.SKM4.640180', '1.SKB8.640193', '1.SKD8.640184',
                      '1.SKM9.640192', '1.SKB7.640196']
        ord_res_fp = self._create_ordination_results(sample_ids)

        # Test success
        obs_success, obs_ainfo, obs_error = _validate_ordination_results(
            {'plain_text': [ord_res_fp]}, self.metadata, self.out_dir)
        self.assertTrue(obs_success)
        exp_ainfo = [ArtifactInfo(None, "ordination_results",
                     [(ord_res_fp, 'plain_text')])]
        self.assertEqual(obs_ainfo, exp_ainfo)
        self.assertEqual(obs_error, "")

        # Test failure
        sample_ids = ['1.SKM4.640180', '1.SKB8.640193', '1.SKD8.640184',
                      '1.SKM9.640192', 'NotASample']
        ord_res_fp = self._create_ordination_results(sample_ids)
        obs_success, obs_ainfo, obs_error = _validate_ordination_results(
            {'plain_text': [ord_res_fp]}, self.metadata, self.out_dir)
        self.assertFalse(obs_success)
        self.assertIsNone(obs_ainfo)
        self.assertEqual(obs_error, "The ordination results contain samples "
                                    "not present in the metadata")

    def test_validate_alpha_vector(self):
        # Create the alpha vector
        sample_ids = ['1.SKM4.640180', '1.SKB8.640193', '1.SKD8.640184',
                      '1.SKM9.640192']
        alpha_vector_fp = self._create_alpha_vector(sample_ids)

        # Test success
        obs_success, obs_ainfo, obs_error = _validate_alpha_vector(
            {'plain_text': [alpha_vector_fp]}, self.metadata, self.out_dir)
        self.assertEqual(obs_error, "")
        self.assertTrue(obs_success)
        exp_ainfo = [ArtifactInfo(None, "alpha_vector",
                     [(alpha_vector_fp, 'plain_text')])]
        self.assertEqual(obs_ainfo, exp_ainfo)

        # Test failure wrong ids
        sample_ids = ['1.SKM4.640180', '1.SKB8.640193', '1.SKD8.640184',
                      'NotASample']
        alpha_vector_fp = self._create_alpha_vector(sample_ids)
        obs_success, obs_ainfo, obs_error = _validate_alpha_vector(
            {'plain_text': [alpha_vector_fp]}, self.metadata, self.out_dir)
        self.assertEqual(obs_error, "The alpha vector contains samples not "
                                    "present in the metadata")
        self.assertFalse(obs_success)
        self.assertIsNone(obs_ainfo)

        # Test failure wrong format
        fd, alpha_vector_fp = mkstemp(suffix='.txt', dir=self.out_dir)
        close(fd)
        with open(alpha_vector_fp, 'w') as f:
            f.write("\tobserved_otus\nsample 1\n")
        obs_success, obs_ainfo, obs_error = _validate_alpha_vector(
            {'plain_text': [alpha_vector_fp]}, self.metadata, self.out_dir)
        self.assertEqual(obs_error, "The alpha vector format is incorrect")
        self.assertFalse(obs_success)
        self.assertIsNone(obs_ainfo)

    def test_validate_sample_data(self):
        # create sample data
        sample_data_fp = join(self.out_dir, 'Mislabeled.txt')
        with open(sample_data_fp, 'w') as f:
            text = [
                'sample_name\tspec\talleged_probability\tMislabeled\t'
                'corrected_label\tSourceSink\tmin_proportion\tContaminated',
                '1788.1\tOsse\t0.91\tFalse\t'
                'Osse\tsource\t0.7818181818181819\tFalse',
                '1788.2\tBubo\t0.91\tFalse\t'
                'Bubo\tsource\t0.8775510204081632\tFalse',
            ]
            f.write('\n'.join(text))
        qza_fp = sample_data_fp.replace('.txt', '.qza')

        obs_success, obs_ainfo, obs_error = _validate_sample_data(
            {'plain_text': [sample_data_fp], 'qza': [qza_fp]},
            self.metadata, self.out_dir)

        self.assertEqual(obs_error, "")
        self.assertTrue(obs_success)

        exp_ainfo = [ArtifactInfo(None, "SampleData",
                     [(sample_data_fp, 'plain_text'), (qza_fp, 'qza')])]
        self.assertEqual(obs_ainfo, exp_ainfo)

        # Test failure:
        obs_success, obs_ainfo, obs_error = _validate_sample_data(
            {'plain_text': [sample_data_fp]}, self.metadata, self.out_dir)
        self.assertEqual(obs_error, "The artifact is missing a QZA file")
        self.assertFalse(obs_success)
        self.assertIsNone(obs_ainfo)

    def test_validate(self):
        # Test artifact type error
        job_id, params = self._create_job(
            'NotAType', {'plan_text': 'Will fail before checking this'}, 1)
        obs_success, obs_ainfo, obs_error = validate(
            self.qclient, job_id, params, self.out_dir)
        self.assertFalse(obs_success)
        self.assertIsNone(obs_ainfo)
        self.assertEqual(
            obs_error, "Unknown artifact type NotAType. Supported types: "
                       "FeatureData, SampleData, alpha_vector, "
                       "distance_matrix, ordination_results")

        # Test missing metadata error - to be fair, I don't know how this error
        # can happen in the live system, but better be safe than sorry
        job_id, params = self._create_job(
            'distance_matrix', {'plan_text': 'Will fail before checking this'},
            None)
        obs_success, obs_ainfo, obs_error = validate(
            self.qclient, job_id, params, self.out_dir)
        self.assertFalse(obs_success)
        self.assertIsNone(obs_ainfo)
        self.assertEqual(
            obs_error, "Missing metadata information")

        # Test distance matrix success
        sample_ids = ['1.SKM4.640180', '1.SKB8.640193', '1.SKD8.640184',
                      '1.SKM9.640192', '1.SKB7.640196']
        dm_fp = self.qclient.push_file_to_central(
            self._create_distance_matrix(sample_ids))
        job_id, params = self._create_job(
            'distance_matrix', {'plain_text': [dm_fp]}, 1)
        obs_success, obs_ainfo, obs_error = validate(
            self.qclient, job_id, params, self.out_dir)
        self.assertTrue(obs_success)
        html_fp = join(self.out_dir, 'index.html')
        exp_ainfo = [ArtifactInfo(None, "distance_matrix",
                                  [(dm_fp, 'plain_text'),
                                   (html_fp, 'html_summary')])]
        self.assertEqual(obs_ainfo, exp_ainfo)
        self.assertEqual(obs_error, "")

        # Test ordination results success
        ord_res_fp = self.qclient.push_file_to_central(
            self._create_ordination_results(sample_ids))
        job_id, params = self._create_job(
            'ordination_results', {'plain_text': [ord_res_fp]}, 1)
        obs_success, obs_ainfo, obs_error = validate(
            self.qclient, job_id, params, self.out_dir)
        self.assertTrue(obs_success)
        html_fp = join(self.out_dir, 'index.html')
        esf_fp = join(self.out_dir, 'emperor_support_files')
        exp_ainfo = [ArtifactInfo(None, "ordination_results",
                     [(ord_res_fp, 'plain_text'),
                      (html_fp, 'html_summary'),
                      (esf_fp, 'html_summary_dir')])]
        self.assertEqual(obs_ainfo, exp_ainfo)
        self.assertEqual(obs_error, "")

        # Test alpha vector success
        alpha_vector_fp = self.qclient.push_file_to_central(
            self._create_alpha_vector(sample_ids))
        job_id, params = self._create_job(
            'alpha_vector', {'plain_text': [alpha_vector_fp]}, 1)
        obs_success, obs_ainfo, obs_error = validate(
            self.qclient, job_id, params, self.out_dir)
        self.assertTrue(obs_success)
        html_fp = join(self.out_dir, 'index.html')
        sf_fp = join(self.out_dir, 'support_files')
        exp_ainfo = [ArtifactInfo(None, "alpha_vector",
                     [(alpha_vector_fp, 'plain_text'),
                      (html_fp, 'html_summary'),
                      (sf_fp, 'html_summary_dir')])]
        self.assertEqual(obs_ainfo, exp_ainfo)
        self.assertEqual(obs_error, "")

    def test_validate_FeatureData(self):
        # Create the feature data
        fd, fd_fp = mkstemp(suffix='.txt', dir=self.out_dir)
        close(fd)
        with open(fd_fp, 'w') as f:
            f.write("Feature ID\tTaxonomy\tConfidence\n")
            f.write("TACGGAGGA\tk__Bacteria;p__Bacteroidetes;c__Bacteroidia\t"
                    "0.9998743\n")
            f.write("TACGTAGGG\tk__Bacteria;p__Firmicutes;c__Clostridia\t"
                    "0.9999999\n")

        # Test success
        obs_success, obs_ainfo, obs_error = _validate_feature_data(
            {'plain_text': [fd_fp]}, None, self.out_dir)
        self.assertEqual(obs_error, "")
        self.assertTrue(obs_success)
        exp_ainfo = [ArtifactInfo(None, "FeatureData",
                     [(fd_fp, 'plain_text')])]
        self.assertEqual(obs_ainfo, exp_ainfo)

        # Test failure wrong format
        fd, fd_fp = mkstemp(suffix='.txt', dir=self.out_dir)
        close(fd)
        with open(fd_fp, 'w') as f:
            f.write("Feature ID\tIt's gonna fail!\tConfidence\n")
            f.write("TACGGAGGA\tk__Bacteria;p__Bacteroidetes;c__Bacteroidia\t"
                    "0.9998743\n")
            f.write("TACGTAGGG\tk__Bacteria;p__Firmicutes;c__Clostridia\t"
                    "0.9999999\n")
        obs_success, obs_ainfo, obs_error = _validate_feature_data(
            {'plain_text': [fd_fp]}, None, self.out_dir)
        self.assertIn("The file header seems wrong", obs_error)
        self.assertFalse(obs_success)
        self.assertIsNone(obs_ainfo)


if __name__ == '__main__':
    main()
